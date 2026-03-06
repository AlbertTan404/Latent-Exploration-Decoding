import argparse, json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir', type=str,
        default='results/model_name/exp_name'
    )

    args, _ = parser.parse_known_args()

    rename_lcb_fn(Path(args.src_dir))


def rename_lcb_fn(src_dir: Path):
    src_path = src_dir / 'livecodebench_results_converted_codegeneration_output_eval.json'
    src_file = json.load(src_path.open('r'))[0]
    pass_at_1 = src_file['pass@1']
    pass_at_all = [float(i > 0) for i in src_file['detail']['pass@1'].values()]
    pass_at_all = sum(pass_at_all) / len(pass_at_all)

    original_fp = list(src_dir.glob('livecodebench-statistics-*.json'))
    if len(original_fp) != 1:
        print(f'Error: {original_fp}')
        return

    original_fp = original_fp[0]

    original_fn = original_fp.stem.removeprefix('livecodebench-statistics-')
    parts = original_fn.split('-')
    if len(parts) != 3:
        print(f'Error: {original_fn}')
        return
    length = int(parts[-1])

    updated_fp = src_dir / f'livecodebench-statistics-{pass_at_1:.4f}-{pass_at_all:.4f}-{length}.json'
    shutil.move(original_fp, updated_fp)

if  __name__ == '__main__':
    main()
