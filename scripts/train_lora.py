import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config.config import AugmentationConfig, LoRAConfig, ModelConfig
from src.data.augmentation import create_noise_augmentation
from src.data.dataset_factory import make_eval_dataset, make_train_dataset
from src.models.lora_setup import apply_lora, create_lora_config, load_processor, load_whisper_model, prepare_model_for_training
from src.training.training_runner import run_training

def main():
    parser = argparse.ArgumentParser(description='Train LoRA model')
    parser.add_argument('--train-manifest', type=str, required=True, help='Train manifest')
    parser.add_argument('--eval-manifest', type=str, required=True, help='Eval manifest')
    parser.add_argument('--model-id', type=str, default=None, help='HuggingFace model ID')
    parser.add_argument('--augmentation', type=str, choices=['none', 'noise'], default='none')
    parser.add_argument('--noise-dir', type=str, default='./noise_data', help='Noise directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/whisper-v3-lora')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--eval-strategy', type=str, choices=['no', 'steps', 'epoch'], default='epoch')
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import torch
    assert torch.cuda.is_available(), 'CUDA is not available'
    if args.augmentation == 'noise' and (not args.noise_dir):
        parser.error('--noise-dir is required')
    model_id = args.model_id or ModelConfig().v3_model_id
    torch_dtype_str = 'bfloat16' if args.bf16 else 'float32'
    print('train lora')
    print(f'Model: {model_id}')
    print(f'Train: {args.train_manifest}')
    print(f'Eval:  {args.eval_manifest}')
    print(f'Augmentation: {args.augmentation}')
    print(f'Output: {args.output_dir}')
    print(f'LoRA r={args.lora_r}, LR={args.learning_rate}, BS={args.batch_size}')
    print(f'Eval strategy: {args.eval_strategy}')
    print(f'Seed: {args.seed}')
    model = load_whisper_model(model_id=model_id, torch_dtype=torch_dtype_str)
    model = model.to('cuda')
    processor = load_processor(model_id=model_id)
    model = prepare_model_for_training(model, use_gradient_checkpointing=False)
    lora_cfg = LoRAConfig(r=args.lora_r)
    lora_config = create_lora_config(lora_cfg)
    model = apply_lora(model, lora_config)
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
    print('\ntraining')
    model, processor = run_training(model=model, processor=processor, train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=args.output_dir, mode='lora', max_steps=args.max_steps, num_train_epochs=args.num_epochs, eval_strategy=args.eval_strategy, per_device_train_batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
    print(f"\nsaved: {os.path.join(args.output_dir, 'lora-adapter')}")
if __name__ == '__main__':
    main()
