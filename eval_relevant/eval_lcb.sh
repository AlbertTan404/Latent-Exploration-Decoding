path=$1

cd /Path/To/LiveCodeBench && python -m lcb_runner.runner.custom_evaluator --custom_output_file $1 --release_version release_v5 --start_date 2024-08-01T00:00:00 --num_process_evaluate 32 --trust_remote_code

exp_dir=$(dirname "$path")
python /Path/To/LED/eval/get_lvb_stats.py --src_dir $exp_dir