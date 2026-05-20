import argparse
import gc
import json
import os
import sys
import torch
from transformers import WhisperForConditionalGeneration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config.config import AugmentationConfig, LoRAConfig
from src.data.augmentation import create_noise_augmentation
from src.data.dataset_factory import attach_eval_transform, attach_train_transform, load_and_filter_dataset
from src.data.dataset_loader import load_manifest_as_dataset
from src.evaluation.evaluator import evaluate_model
from src.models.lora_setup import load_processor, load_whisper_model, prepare_model_for_training, setup_model_for_lora
from src.training.training_runner import run_training
ALL_BASELINES = ['zero_shot', 'full_ft_clean', 'lora_clean', 'full_ft_noisy', 'lora_noisy', 'lora_noisy_r32']
SNR_LEVELS = [0, 5, 10, 15, 20]

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run ASR experiment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group('Data')
    g.add_argument('--train-manifest', type=str, required=True, help='Train manifest path')
    g.add_argument('--eval-manifest', type=str, required=True, help='Eval manifest path')
    g.add_argument('--noise-dir', type=str, default='/root/DisserRealization/noise_data', help='Noise data directory')
    g.add_argument('--eval-root', type=str, default='/root/DisserRealization/eval/noisy_test_sets', help='Test set directory')
    g.add_argument('--farfield-manifest', type=str, default=None, help='Farfield manifest path')
    g2 = p.add_argument_group('Limits')
    g2.add_argument('--train-limit', type=int, default=None, help='Train sample limit')
    g2.add_argument('--eval-limit', type=int, default=None, help='Eval sample limit')
    g2.add_argument('--test-limit', type=int, default=None, help='Test sample limit')
    g3 = p.add_argument_group('Training')
    g3.add_argument('--model-id', type=str, default='openai/whisper-large-v3')
    g3.add_argument('--max-steps', type=int, default=-1, help='Max steps')
    g3.add_argument('--num-epochs', type=int, default=3, help='Epoch count')
    g3.add_argument('--lora-bs', type=int, default=24, help='LoRA batch size')
    g3.add_argument('--full-bs', type=int, default=16, help='Full fine-tune batch size')
    g3.add_argument('--lora-ga', type=int, default=1, help='LoRA gradient accumulation')
    g3.add_argument('--full-ga', type=int, default=4, help='Full fine-tune gradient accumulation')
    g3.add_argument('--lora-lr', type=float, default=0.001)
    g3.add_argument('--full-lr', type=float, default=1e-05)
    g3.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    g3.add_argument('--eval-strategy', type=str, choices=['no', 'steps', 'epoch'], default='epoch', help='Eval strategy')
    g3.add_argument('--seed', type=int, default=42, help='Random seed')
    g4 = p.add_argument_group('Baselines')
    g4.add_argument('--baselines', nargs='+', default=ALL_BASELINES, choices=ALL_BASELINES, metavar='BASELINE', help='Baselines to run')
    g4.add_argument('--force-rerun', action='store_true', help='Overwrite existing results')
    g5 = p.add_argument_group('Output')
    g5.add_argument('--output-root', type=str, default='/root/DisserRealization/checkpoints')
    g5.add_argument('--results-dir', type=str, default='/root/DisserRealization/eval/results')
    return p

def load_test_sets(eval_root: str, test_limit: int | None, farfield_manifest: str | None=None) -> dict:
    sets = {}
    clean_path = os.path.join(eval_root, 'clean', 'manifest.jsonl')
    sets['clean'] = load_manifest_as_dataset(clean_path, limit=test_limit)
    for snr in SNR_LEVELS:
        name = f'snr_{snr}db'
        path = os.path.join(eval_root, name, 'manifest.jsonl')
        sets[name] = load_manifest_as_dataset(path, limit=test_limit)
    if farfield_manifest and os.path.exists(farfield_manifest):
        sets['farfield'] = load_manifest_as_dataset(farfield_manifest, limit=test_limit)
    elif farfield_manifest:
        raise FileNotFoundError(f'missing farfield manifest: {farfield_manifest}')
    return sets

def _noise_categories(noise_dir: str) -> dict[str, str]:
    splits_train = os.path.join(noise_dir, 'splits', 'train')
    if os.path.isdir(splits_train):
        return {'musan_noise': os.path.join(splits_train, 'musan_noise'), 'musan_music': os.path.join(splits_train, 'musan_music'), 'musan_speech': os.path.join(splits_train, 'musan_speech'), 'esc50': os.path.join(splits_train, 'esc50'), 'demand': os.path.join(splits_train, 'demand')}
    raise FileNotFoundError(f'missing noise split: {splits_train}')

def _make_augmenter(noise_dir: str, seed: int=42):
    cats = _noise_categories(noise_dir)
    missing = [v for v in cats.values() if not os.path.isdir(v)]
    if missing:
        raise FileNotFoundError(f'missing noise directories: {missing}')
    return create_noise_augmentation(noise_paths=cats, config=AugmentationConfig(), seed=seed)

def _eval_all(model, processor, test_sets: dict, label: str, device, dtype) -> dict:
    results = {}
    for test_name, ds in test_sets.items():
        m = evaluate_model(model, processor, ds, device=device, dtype=dtype, desc=f'{label} | {test_name}')
        results[test_name] = m
        print(f"    {test_name}: WER={m['wer']:.2f}%  CER={m['cer']:.2f}%")
    return results

def run_zero_shot(args, test_sets, processor, device, dtype) -> dict:
    print('\nbaseline: zero_shot')
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype)
    results = _eval_all(model, processor, test_sets, 'zero_shot', device, dtype)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

def run_full_ft(args, test_sets, processor, train_base, eval_base, noisy: bool, device, dtype) -> dict:
    label = 'full_ft_noisy' if noisy else 'full_ft_clean'
    print(f'\nbaseline: {label}')
    model = load_whisper_model(model_id=args.model_id, torch_dtype='bfloat16')
    model = model.to(device)
    model = prepare_model_for_training(model, use_gradient_checkpointing=False)
    augmenter = _make_augmenter(args.noise_dir, seed=args.seed) if noisy else None
    train_ds = attach_train_transform(train_base, processor, augmenter)
    eval_ds = attach_eval_transform(eval_base, processor)
    max_steps, num_epochs = _resolve_steps(args)
    model, processor = run_training(model, processor, train_ds, eval_ds, output_dir=os.path.join(args.output_root, label), mode='full', max_steps=max_steps, num_train_epochs=num_epochs, eval_strategy=args.eval_strategy, per_device_train_batch_size=args.full_bs, gradient_accumulation_steps=args.full_ga, learning_rate=args.full_lr, seed=args.seed)
    results = _eval_all(model, processor, test_sets, label, device, dtype)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

def run_lora(args, test_sets, processor, train_base, eval_base, noisy: bool, device, dtype, label: str=None) -> dict:
    if label is None:
        label = 'lora_noisy' if noisy else 'lora_clean'
    print(f'\nbaseline: {label}')
    lora_cfg = LoRAConfig(r=args.lora_r)
    model, _proc_unused = setup_model_for_lora(model_id=args.model_id, torch_dtype='bfloat16', lora_config=lora_cfg, use_gradient_checkpointing=False)
    model = model.to(device)
    augmenter = _make_augmenter(args.noise_dir, seed=args.seed) if noisy else None
    train_ds = attach_train_transform(train_base, processor, augmenter)
    eval_ds = attach_eval_transform(eval_base, processor)
    max_steps, num_epochs = _resolve_steps(args)
    model, processor = run_training(model, processor, train_ds, eval_ds, output_dir=os.path.join(args.output_root, label), mode='lora', max_steps=max_steps, num_train_epochs=num_epochs, eval_strategy=args.eval_strategy, per_device_train_batch_size=args.lora_bs, gradient_accumulation_steps=args.lora_ga, learning_rate=args.lora_lr, seed=args.seed)
    results = _eval_all(model, processor, test_sets, label, device, dtype)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

def _resolve_steps(args) -> tuple[int, int | None]:
    if args.max_steps > 0:
        return (args.max_steps, None)
    return (-1, args.num_epochs)

def print_summary(all_results: dict):
    standard = ['clean'] + [f'snr_{s}db' for s in SNR_LEVELS]
    present_extra = [k for k in ['farfield'] if any((k in res for res in all_results.values()))]
    test_names = standard + present_extra
    col_w = 12
    print('\nWER summary')
    header = f"{'Baseline':<22}" + ''.join((f'{t:>{col_w}}' for t in test_names))
    print(header)
    print('-' * len(header))
    for name, res in all_results.items():
        row = f'{name:<22}'
        for t in test_names:
            row += f"{res[t]['wer']:>{col_w - 1}.2f}%" if t in res else f"{'-':>{col_w}}"
        print(row)

def run(args):
    if not torch.cuda.is_available():
        print('CUDA is not available', file=sys.stderr)
        sys.exit(1)
    device = 'cuda'
    dtype = torch.bfloat16
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    print('experiment')
    print(f'  model:      {args.model_id}')
    print(f'  baselines:  {args.baselines}')
    print(f'  Train:       {args.train_manifest}' + (f'  (limit={args.train_limit})' if args.train_limit else ''))
    print(f'  Noise dir:   {args.noise_dir}')
    steps_str = f'max_steps={args.max_steps}' if args.max_steps > 0 else f'epochs={args.num_epochs}'
    print(f'  schedule:    {steps_str}')
    print(f'  eval:        {args.eval_strategy}')
    print(f'  seed:        {args.seed}')
    farfield = getattr(args, 'farfield_manifest', None)
    print(f"  farfield:    {farfield or '-'}")
    print('\nloading test sets')
    test_sets = load_test_sets(args.eval_root, args.test_limit, farfield_manifest=farfield)
    for name, ds in test_sets.items():
        print(f'  {name}: {len(ds)}')
    print('\nloading train/eval data')
    processor = load_processor(model_id=args.model_id)
    train_base = load_and_filter_dataset(args.train_manifest, processor, limit=args.train_limit)
    eval_base = load_and_filter_dataset(args.eval_manifest, processor, limit=args.eval_limit)
    print(f'  train: {len(train_base)}')
    print(f'  eval:  {len(eval_base)}')
    all_results: dict = {}
    dispatch = {'zero_shot': lambda: run_zero_shot(args, test_sets, processor, device, dtype), 'full_ft_clean': lambda: run_full_ft(args, test_sets, processor, train_base, eval_base, noisy=False, device=device, dtype=dtype), 'lora_clean': lambda: run_lora(args, test_sets, processor, train_base, eval_base, noisy=False, device=device, dtype=dtype), 'full_ft_noisy': lambda: run_full_ft(args, test_sets, processor, train_base, eval_base, noisy=True, device=device, dtype=dtype), 'lora_noisy': lambda: run_lora(args, test_sets, processor, train_base, eval_base, noisy=True, device=device, dtype=dtype), 'lora_noisy_r32': lambda: run_lora(args, test_sets, processor, train_base, eval_base, noisy=True, device=device, dtype=dtype, label='lora_noisy_r32')}
    for name in args.baselines:
        out = os.path.join(args.results_dir, f'{name}.json')
        if os.path.exists(out) and (not args.force_rerun):
            with open(out, 'r', encoding='utf-8') as f:
                all_results[name] = json.load(f)
            print(f'\n[skip] {name}: {out}')
            continue
        all_results[name] = dispatch[name]()
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, ensure_ascii=False, indent=2)
    print_summary(all_results)
    summary_path = os.path.join(args.results_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'\nresults: {args.results_dir}')

def main():
    args = get_parser().parse_args()
    run(args)
if __name__ == '__main__':
    main()
