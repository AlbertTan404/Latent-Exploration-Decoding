# From https://github.com/eric-ai-lab/Soft-Thinking
import json
from pathlib import Path
import argparse
import re


def extract_code( text: str) -> str:
    # Use regex to find the content inside ```python\n...\n```
    matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    # Return the last match if any matches exist
    completion_code = matches[-1] if matches else ""
    return completion_code


def convert_json(input_file: str):
    input_file = Path(input_file)
    output_file = input_file.parent / f"{input_file.stem}_converted.json"

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # 初始化结果列表
    result = []

    # 遍历数据并处理
    for item in data:
        question_id = item.get("src_item").get('final_answer').get("question_id")
        completion = item.get("completion_texts")

        # 如果 completion 是字符串，转换为单元素列表
        if isinstance(completion, str):
            completion = [extract_code(completion)]
        else:
            completion = [extract_code(c) for c in completion]

        # 构造新的字典
        result.append({"question_id": question_id, "code_list": completion})

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Convert JSON file format for LiveCodeBench.")
    parser.add_argument("--input_file", type=str, default="results/Qwen3-4B-Thinking-2507/cot_0.0_0.95_20_0_1.0_32768/livecodebench_results.json", help="Path to the input JSON file.")

    args, _ = parser.parse_known_args()

    # 调用转换函数
    convert_json(args.input_file)
