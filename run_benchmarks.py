from typing import List
import os
import json
import numpy as np
import time
from pathlib import Path
import argparse
from dataclasses import asdict
from transformers import AutoTokenizer
from transformers import AutoConfig

import sglang as sgl

from eval_relevant import matheval
from eval_relevant.convert_livecodebench import convert_json
from led.decoding import LatentDecodingParams
from led.utils import is_debug_mode, get_timestamp, safe_mean
from led.constants import MODEL_PATH_DICT, MATH_DATASETS, CODE_DATASETS, PROMPT


def load_dataset(dataset):
    if os.path.exists(f"./datasets/{dataset}.json"):
        with open(f"./datasets/{dataset}.json") as f:
            samples = json.load(f)
    else:
        raise ValueError("Invalid dataset name")
    
    if is_debug_mode():
        samples = samples[:4]

    for micro_index, sample in enumerate(samples):
        sample['dataset'] = dataset
        sample['micro_index'] = micro_index
    
    return samples


def get_args():
    parser = argparse.ArgumentParser(description='Process some parameters for text generation.')

    #++ general arguments
    parser.add_argument('--datasets', type=str, default='aime2024,aime2025,gsm8k,math500,gpqa_diamond,livecodebench')
    parser.add_argument('--sampling_backend', type=str, choices=["flashinfer"], default="flashinfer", help='Sampling backend')
    parser.add_argument('--model_name', type=str, default="Qwen3-4B-Thinking-2507", help='Model name or path')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tensor parallel size, tp_size)')
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None, help='Max number of batch runned in one time.')
    parser.add_argument('--max_running_requests', type=int, default=512, help='Max number of requests runned together.')
    parser.add_argument('--max_batch', type=int, default=1000000, help='Max number of batch runned in one time.')
    parser.add_argument('--mem_fraction_static', type=float, default=0.75, help='Max memory to use per gpu.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=16, help='Sampling number')
    parser.add_argument('--max_generated_tokens', type=int, default=32768, help='Limit the number of generated tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling probability')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k sampling probability')
    parser.add_argument('--min_p', type=float, default=0, help='Min-p sampling probability')
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6, help='Temperature after thinking')
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95, help='Top-p after thinking')
    parser.add_argument('--after_thinking_top_k', type=int, default=20, help='Top-k after thinking')
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0, help='Min-p after thinking')
    #--

    #++ SoftThinking parameters 
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.01, help='Early stopping entropy threshold (set it to 0.0 to disable early stopping)')
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256, help='Early stopping length threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0, help='Dirichlet alpha')
    parser.add_argument('--gumbel_softmax_temperature', type=float, default=0.5, help='Gumbel-softmax temperature')
    parser.add_argument('--add_noise_dirichlet', action='store_true', help='Add Dirichlet noise to sampling')
    parser.add_argument('--add_noise_gumbel_softmax', action='store_true', help='Add Gumbel-softmax noise to sampling')
    parser.add_argument("--enable_soft_thinking", action="store_true", help="Enable soft thinking mode")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=10)
    #--

    #++ LED & DoLa relevant parameters
    parser.add_argument("--latent_method", type=str, default="")
    parser.add_argument("--think_only", type=int, default=1)
    parser.add_argument("--ln", type=int, default=0)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument('--do_exploit', type=int, default=1)
    parser.add_argument("--sub_method", type=str, default="")
    #--

    #++ utils
    parser.add_argument('--log_suffix', type=str, default="")
    parser.add_argument('--signature', type=str, default="")
    parser.add_argument('--debug', action='store_true')
    #--

    args = parser.parse_args()

    print(f'Arguments: {args}')

    if args.model_name == 'MiMo-7B-RL':  # limited to 32k
        args.max_generated_tokens = min(28*1024, args.max_generated_tokens)  # max prefix length ~= 3k

    if is_debug_mode() or args.debug:
        args.output_dir = f'./temp/{get_timestamp()}'
    
    return args


def main():
    args = get_args()

    datasets = args.datasets.split(',')
    model_path = MODEL_PATH_DICT[args.model_name]
    max_generated_tokens = args.max_generated_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p
    think_end_str = args.think_end_str
    random_seed = args.random_seed
    num_gpus = args.num_gpus
    max_running_requests = args.max_running_requests
    mem_fraction_static = args.mem_fraction_static
    
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params = {
        "temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": min_p, "repetition_penalty": args.repetition_penalty,
        "after_thinking_temperature": args.after_thinking_temperature, "after_thinking_top_p": args.after_thinking_top_p, "after_thinking_top_k": args.after_thinking_top_k, "after_thinking_min_p": args.after_thinking_min_p,
        "n": 1, # repeat prompt for num_samples times instead of using num_samples in sampling_params
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature, "dirichlet_alpha": args.dirichlet_alpha,
        "max_new_tokens": max_generated_tokens, "think_end_str": think_end_str,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold
    }

    decoding_params = LatentDecodingParams(
        method=args.latent_method.lower(),
        think_only=args.think_only,
        d=args.d,
        k=args.k,
        ln=args.ln,
        sub_method=args.sub_method,
        do_exploit=args.do_exploit,
    )
    print(f"Arguments: {args}", flush=True)
    print(f"Decoding params: {decoding_params}", flush=True)

    if args.latent_method:
        if decoding_params.method == 'dola':
            dola_layers = decoding_params.sub_method
            exp_name = f"dola_{dola_layers}_{temperature}_{top_p}_{top_k}_{min_p}_{args.repetition_penalty}_{max_generated_tokens}"

            #++ From transformers.generation.utils.GenerationMixin._dola_decoding
            final_layer = model_config.num_hidden_layers
            if not model_config.tie_word_embeddings:
                start_layer = 0
            elif final_layer > 2:
                start_layer = 2
            elif final_layer == 2:
                start_layer = 1
            else:
                start_layer = 0
            
            if dola_layers == "low":  # better for longer sequences (reasoning)
                if start_layer == final_layer // 2:
                    candidate_premature_layers = [start_layer]
                else:
                    candidate_premature_layers = (
                        list(range(start_layer, final_layer // 2, 2))
                        if final_layer <= 40
                        else list(range(start_layer, 20, 2))
                    )
            elif dola_layers == "high":  # better for shorter sequences
                candidate_premature_layers = (
                    list(range(final_layer // 2, final_layer, 2))
                    if final_layer <= 40
                    else list(range(final_layer - 20, final_layer, 2))
                )
            else:
                raise ValueError("Invalid dola_layers")
            #--

            decoding_params.candidate_premature_layers = candidate_premature_layers
        else:  # LED
            decoding_params_dict = asdict(decoding_params)
            exp_name = f"led-{temperature:.1f}-" + "-".join([f"{k}_{v}" for k, v in decoding_params_dict.items()])

    elif args.enable_soft_thinking:
        noise_suffix = (
            (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
            + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
        )
        exp_name = (
            f"softthinking_"
            f"{temperature}_{top_p}_{top_k}_{min_p}_{args.repetition_penalty}_{args.dirichlet_alpha}_"
            f"{args.max_topk}_{max_generated_tokens}_{args.early_stopping_entropy_threshold}_"
            f"{args.early_stopping_length_threshold}{noise_suffix}"
        )
    else:
        exp_name = f"cot_{temperature}_{top_p}_{top_k}_{min_p}_{args.repetition_penalty}_{max_generated_tokens}"

    exp_name = exp_name + args.log_suffix
    
    exp_dir = Path(args.output_dir) / args.model_name / exp_name
    if exp_dir.exists():
        existing_runs = [p.stem.split('_results')[0] for p in exp_dir.glob("*_results.json")]
        datasets = list(set(datasets) - set(existing_runs))
    exp_dir.mkdir(parents=True, exist_ok=True)

    if len(datasets) == 0:
        return

    Path('temp').mkdir(exist_ok=True)
    with open(f'temp/{args.signature}.txt', 'w') as f:
        f.write(exp_name)

    src_items = []
    for dataset in datasets:
        src_items += load_dataset(dataset)

    # prepare prompts
    prompt_list = []
    for src_item in src_items:
        dataset = src_item["dataset"]
        if dataset in ["aime2024", "aime2025", "math500", "gsm8k"]:
            chat = [{"role": "user", "content": PROMPT.MATH_QUERY_TEMPLATE.format(Question=src_item["prompt"][0]["value"])}]
        elif dataset == "gpqa_diamond":
            chat = [{"role": "user", "content": PROMPT.MQA_QUERY_TEMPLATE.format(Question=src_item["prompt"][0]["value"])}]
        elif dataset == "livecodebench":
            chat = [{"role": "user", "content": PROMPT.get_lcb_prompt(question_content=src_item["prompt"][0]["value"], starter_code=src_item["final_answer"]["starter_code"])}]
        else:
            raise ValueError("Invalid dataset name")

        prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        prompt_list.extend([prompt] * args.num_samples)

    # generate results
    print("start", flush=True)
    start_time = time.time()
    print(f"Number of GPUs available: {num_gpus}", flush=True)
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=num_gpus,
        log_level="info",
        trust_remote_code=True,
        random_seed=random_seed,
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        enable_soft_thinking=args.enable_soft_thinking,
        add_noise_dirichlet=args.add_noise_dirichlet,
        add_noise_gumbel_softmax=args.add_noise_gumbel_softmax,
        max_topk=args.max_topk,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        sampling_backend=args.sampling_backend,
        decoding_params=decoding_params,
        enable_think_only=bool(args.think_only),
    )
    outputs = llm.generate(prompt_list, sampling_params)
    end_time = time.time()
    time_taken = end_time - start_time
    print("end", flush=True)
    print(f"Time taken: {time_taken} seconds", flush=True)
    llm.shutdown()

    completion_text_list = [o["text"] for o in outputs]
    finished_list = [o["meta_info"]["finish_reason"]["type"] == "stop" for o in outputs]
    n_tokens_list = [o["meta_info"]["completion_tokens"] for o in outputs]

    backup = {
        "src_items": src_items,
        "completion_text_list": completion_text_list,
        "finished_list": finished_list,
        "n_tokens_list": n_tokens_list,
        "datasets": datasets,
    }

    with open(exp_dir / f"{exp_name}_all_src_data.json", "w") as f:
        json.dump(backup, f)

    # evaluate results
    benchmark_results = {dataset: [] for dataset in datasets}
    for question_idx in range(len(src_items)):
        # print(idx, flush=True)
        src_item = src_items[question_idx]
        dataset = src_item['dataset']

        accuracies = []
        completion_texts = completion_text_list[question_idx * args.num_samples: (question_idx + 1) * args.num_samples]
        finished = finished_list[question_idx * args.num_samples: (question_idx + 1) * args.num_samples]
        thinking_finished = [finished[j] and completion_texts[j].find(think_end_str) != -1 for j in range(args.num_samples)]
        n_tokens = n_tokens_list[question_idx * args.num_samples: (question_idx + 1) * args.num_samples]
        n_think_tokens = [len(tokenizer.tokenize(completion_texts[j][:completion_texts[j].find(think_end_str)])) for j in range(args.num_samples)]
        extracted_answers = []

        # evaluate each sample
        for j in range(args.num_samples):
            if dataset == "livecodebench":
                accuracies.append(0.0)
                extracted_answers.append("")
            try:
                rule_judge_result, extracted_answer = matheval.evaluator_map[dataset].rule_judge(
                    completion_texts[j], src_item["final_answer"], finished[j]
                )
                accuracies.append(1.0 if rule_judge_result else 0.0)
                extracted_answers.append(extracted_answer)
            except Exception as e:
                accuracies.append(0.0)
                extracted_answers.append(completion_texts[j])
                # print(f"Failed to evaluate completion text {completion_texts[j]}: {str(e)}", flush=True)

        result_item = {
            "hyperparams": str(args),
            "src_item": src_item,
            "completion_texts": completion_texts,
            "extracted_answers": extracted_answers,
            "accuracies": accuracies,
            "pass@k": np.max(accuracies),
            "n_tokens": n_tokens,
            "n_think_tokens": n_think_tokens,
            "avg_n_tokens": np.mean(n_tokens),
            "avg_n_think_tokens": np.mean(n_think_tokens),
            "finished": finished,
            "thinking_finished": thinking_finished,
        }
        benchmark_results[dataset].append(result_item)

    # save results
    with (exp_dir / f'time_taken_for_{datasets}_{time_taken:.2f}s.txt').open('w') as f:
        f.write(str(args) + '\n')
        f.write(f"Time taken for {datasets}: {time_taken}")

    for dataset in datasets:
        result_item_list = benchmark_results[dataset]
        statistics = []
        results_save_path = exp_dir / f'{dataset}_results.json'
        with results_save_path.open('w') as f:
            json.dump(result_item_list, f, indent=2)
        if dataset == 'livecodebench':
            convert_json(results_save_path)
        
        all_accuracies = []
        all_n_tokens = []
        all_n_think_tokens = []
        all_n_tokens_finished = []
        all_n_think_tokens_finished = []
        all_n_tokens_correct = []
        all_n_think_tokens_correct = []
        pass_at_k = []
        for result_item in result_item_list:
            all_accuracies.extend(result_item["accuracies"])
            all_n_tokens.extend(result_item["n_tokens"])
            all_n_tokens_finished.extend([result_item["n_tokens"][i] for i in range(len(result_item["finished"])) if result_item["finished"][i]])
            all_n_tokens_correct.extend([result_item["n_tokens"][i] for i in range(len(result_item["accuracies"])) if result_item["accuracies"][i] == 1])
            all_n_think_tokens.extend(result_item["n_think_tokens"])
            all_n_think_tokens_finished.extend([result_item["n_think_tokens"][i] for i in range(len(result_item["finished"])) if result_item["finished"][i]])
            all_n_think_tokens_correct.extend([result_item["n_think_tokens"][i] for i in range(len(result_item["accuracies"])) if result_item["accuracies"][i] == 1])
            pass_at_k.append(result_item["pass@k"])
        
        statistics = {
            "question_num": len(result_item_list),
            "finished_num": sum([sum(result_item["finished"]) for result_item in result_item_list]),
            "thinking_finished_num": sum([sum(result_item["thinking_finished"]) for result_item in result_item_list]),
            "avg_accuracy": safe_mean(all_accuracies),
            "pass@k": safe_mean(pass_at_k),
            "avg_token_length-all": safe_mean(all_n_tokens),
            "avg_token_length-finished": safe_mean(all_n_tokens_finished),
            "avg_token_length-correct": safe_mean(all_n_tokens_correct),
            "avg_think_token_length-all": safe_mean(all_n_think_tokens),
            "avg_think_token_length-finished": safe_mean(all_n_think_tokens_finished),
            "avg_think_token_length-correct": safe_mean(all_n_think_tokens_correct),
        }
        avg_acc = statistics["avg_accuracy"]
        avg_n = statistics["avg_token_length-all"]
        avg_passk = statistics["pass@k"]

        all_passk = sorted([(result_item['src_item']["micro_index"], result_item["pass@k"]) for result_item in result_item_list], key=lambda x: x[0])
        statistics["all_pass@k"] = {i:j for i,j in all_passk}

        with (exp_dir / f'{dataset}-statistics-{avg_acc:.4f}-{avg_passk:.4f}-{int(avg_n)}.json').open('w') as f:
            json.dump(statistics, f, indent=2)
    

if __name__ == "__main__":
    main()
