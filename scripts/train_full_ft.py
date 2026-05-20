import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config.config import AugmentationConfig, ModelConfig
from src.data.augmentation import create_noise_augmentation
from src.data.dataset_factory import make_eval_dataset, make_train_dataset
from src.models.lora_setup import load_processor, load_whisper_model, prepare_model_for_training
from src.training.training_runner import run_training

def main():
    parser = argparse.ArgumentParser(description='Full Fine-Tuning Whisper')
    parser.add_argument('--train-manifest', type=str, required=True, help='Train manifest')
    parser.add_argument('--eval-manifest', type=str, required=True, help='Eval manifest')
    parser.add_argument('--model-id', type=str, default=None, help='HuggingFace model ID')
    parser.add_argument('--augmentation', type=str, choices=['none', 'noise'], default='none')
    parser.add_argument('--noise-dir', type=str, default='./noise_data', help='Noise directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/whisper-v3-full-ft')
    parser.add_argument('--learning-rate', type=float, default=1e-05)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--eval-strategy', type=str, choices=['no', 'steps', 'epoch'], default='epoch')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import torch
    assert torch.cuda.is_available(), 'CUDA is not available'
    model_id = args.model_id or ModelConfig().v3_model_id
    label = 'clean' if args.augmentation == 'none' else 'noisy'
    print(f'train full ({label})')
    print(f'Model: {model_id}')
    print(f'Train: {args.train_manifest}')
    print(f'Eval:  {args.eval_manifest}')
    print(f'Output: {args.output_dir}')
    print(f'LR={args.learning_rate}, BS={args.batch_size}, GA={args.gradient_accumulation}')
    print(f'Effective batch size: {args.batch_size * args.gradient_accumulation}')
    print(f'Eval strategy: {args.eval_strategy}')
    print(f'Seed: {args.seed}')
    model = load_whisper_model(model_id=model_id, torch_dtype='bfloat16')
    model = model.to('cuda')
    processor = load_processor(model_id=model_id)
    model = prepare_model_for_training(model, use_gradient_checkpointing=False)
    print(f'Parameters: {sum((p.numel() for p in model.parameters())):,}')
    augmenter = None
    if args.augmentation == 'noise':
        splits_train = os.path.join(args.noise_dir, 'splits', 'train')
        if os.path.isdir(splits_train):
            noise_paths = [os.path.join(splits_train, 'musan_noise'), os.path.join(splits_train, 'musan_music'), os.path.join(splits_train, 'musan_speech'), os.path.join(splits_train, 'esc50'), os.path.join(splits_train, 'demand')]
        else:
            raise FileNotFoundError(f'missing noise split: {splits_train}')
        for path in noise_paths:
            if not os.path.isdir(path):
                raise FileNotFoundError(f'missing noise directory: {path}')
        augmenter = create_noise_augmentation(noise_paths=noise_paths, config=AugmentationConfig(), seed=args.seed)
        print(f'Noise dirs: {len(noise_paths)}')
    train_dataset = make_train_dataset(args.train_manifest, processor, augmenter=augmenter)
    eval_dataset = make_eval_dataset(args.eval_manifest, processor)
    print(f'Train: {len(train_dataset)} | Eval: {len(eval_dataset)}')
    print(f'\ntraining: {label}')
    model, processor = run_training(model=model, processor=processor, train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=args.output_dir, mode='full', max_steps=args.max_steps, num_train_epochs=args.num_epochs, eval_strategy=args.eval_strategy, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate, seed=args.seed)
    print(f'\nsaved: {args.output_dir}')
if __name__ == '__main__':
    main()
