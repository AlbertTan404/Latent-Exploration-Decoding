MODEL_PATH_DICT = {
    'Qwen3-4B-Instruct-2507': '/PATH/TO/MODEL',
    'Qwen3-4B-Thinking-2507': 'Qwen/Qwen3-4B-Thinking-2507',
    'MiMo-7B-RL': '/PATH/TO/MODEL',
    'DeepSeek-R1-Distill-Llama-8B': '/PATH/TO/MODEL',
    'Qwen3-30B-A3B-Thinking-2507': 'Qwen/Qwen3-30B-A3B-Thinking-2507',
    'QwQ-32B': '/PATH/TO/MODEL',
}


MATH_DATASETS = ["aime2024", "aime2025", "gsm8k", "math500", "gpqa_diamond"]
CODE_DATASETS = ["livecodebench"]


class PROMPT:
    # prompt templates
    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

    MQA_QUERY_TEMPLATE = """
Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,"answer": "C".

{Question}
""".strip()

    @staticmethod
    def get_lcb_prompt(question_content, starter_code):
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        prompt += f"Question: {question_content}\n\n"
        if starter_code:
            prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
        return prompt
