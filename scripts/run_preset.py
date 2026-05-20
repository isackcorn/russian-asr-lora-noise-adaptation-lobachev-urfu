import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from run_experiment import get_parser, run
ROOT = '/root/DisserRealization'
PRESETS = {'smoke': {'train_manifest': f'{ROOT}/data/mini_train/manifest.jsonl', 'eval_manifest': f'{ROOT}/data/mini_test/manifest.jsonl', 'noise_dir': f'{ROOT}/noise_data', 'eval_root': f'{ROOT}/eval/noisy_test_sets', 'output_root': f'{ROOT}/checkpoints_smoke', 'results_dir': f'{ROOT}/eval/results_smoke', 'max_steps': 10, 'num_epochs': 1, 'train_limit': 10, 'eval_limit': 5, 'test_limit': 5, 'lora_bs': 4, 'full_bs': 2, 'lora_ga': 1, 'full_ga': 1, 'eval_strategy': 'no'}, 'quick': {'train_manifest': f'{ROOT}/data/mini_train/manifest.jsonl', 'eval_manifest': f'{ROOT}/data/mini_test/manifest.jsonl', 'noise_dir': f'{ROOT}/noise_data', 'eval_root': f'{ROOT}/eval/noisy_test_sets', 'output_root': f'{ROOT}/checkpoints_quick', 'results_dir': f'{ROOT}/eval/results_quick', 'max_steps': 300, 'num_epochs': 1, 'train_limit': None, 'eval_limit': None, 'test_limit': None, 'lora_bs': 16, 'full_bs': 4, 'lora_ga': 2, 'full_ga': 4, 'eval_strategy': 'no'}}

def main():
    p = argparse.ArgumentParser(description='Run preset')
    p.add_argument('preset', choices=list(PRESETS), help='Preset name')
    p.add_argument('--baselines', nargs='+', default=None, help='Baselines to run')
    p.add_argument('--model-id', type=str, default=None)
    cli = p.parse_args()
    base_parser = get_parser()
    args = base_parser.parse_args(['--train-manifest', 'placeholder', '--eval-manifest', 'placeholder'])
    preset = PRESETS[cli.preset]
    for key, value in preset.items():
        setattr(args, key, value)
    if cli.baselines:
        args.baselines = cli.baselines
    if cli.model_id:
        args.model_id = cli.model_id
    print(f'preset: {cli.preset}')
    run(args)
if __name__ == '__main__':
    main()
