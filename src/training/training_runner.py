import os
from typing import Optional
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from src.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.utils.metrics import create_compute_metrics

def run_training(model, processor, train_dataset, eval_dataset, output_dir: str, mode: str='lora', max_steps: int=-1, num_train_epochs: Optional[int]=None, eval_strategy: str='no', per_device_train_batch_size: Optional[int]=None, gradient_accumulation_steps: Optional[int]=None, learning_rate: Optional[float]=None, seed: int=42, gradient_checkpointing: bool=False, dataloader_num_workers: int=8) -> tuple:
    model_dtype = next(model.parameters()).dtype
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id, model_dtype=model_dtype)
    compute_metrics = create_compute_metrics(processor)
    if mode == 'lora':
        per_device_bs = per_device_train_batch_size or 32
        gradient_accumulation = gradient_accumulation_steps or 2
        lr = learning_rate or 0.001
    else:
        per_device_bs = per_device_train_batch_size or 8
        gradient_accumulation = gradient_accumulation_steps or 4
        lr = learning_rate or 1e-05
    resolved_epochs = num_train_epochs if num_train_epochs is not None else 1
    resolved_max_steps = max_steps if num_train_epochs is None else -1
    use_best_checkpoint = eval_strategy != 'no' and eval_dataset is not None
    if use_best_checkpoint:
        ckpt_save_strategy = 'epoch'
        load_best = True
        save_total_limit = 2
    else:
        ckpt_save_strategy = 'no'
        load_best = False
        save_total_limit = None
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir, per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=gradient_accumulation, learning_rate=lr, weight_decay=0.01, warmup_ratio=0.05, num_train_epochs=resolved_epochs, max_steps=resolved_max_steps, lr_scheduler_type='cosine', bf16=next(model.parameters()).dtype == torch.bfloat16, gradient_checkpointing=gradient_checkpointing, optim='adamw_torch_fused', eval_strategy=eval_strategy, save_strategy=ckpt_save_strategy, save_total_limit=save_total_limit, load_best_model_at_end=load_best, metric_for_best_model='wer', greater_is_better=False, per_device_eval_batch_size=per_device_bs, predict_with_generate=True, generation_max_length=448, logging_steps=50, report_to=['tensorboard'], remove_unused_columns=False, label_names=['labels'], seed=seed, data_seed=seed, dataloader_num_workers=dataloader_num_workers)
    trainer = Seq2SeqTrainer(args=training_args, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics, processing_class=processor)
    duration_str = f'epochs={num_train_epochs}' if num_train_epochs is not None else f'steps={max_steps}'
    best_str = ', best_by=wer' if use_best_checkpoint else ''
    print(f'\ntraining ({mode}, {duration_str}, BS={per_device_bs}, GA={gradient_accumulation}, LR={lr}{best_str})')
    resume_from = None
    if use_best_checkpoint and os.path.isdir(output_dir):
        ckpts = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-') and d.split('-')[-1].isdigit()]
        if ckpts:
            latest = max(ckpts, key=lambda d: int(d.split('-')[-1]))
            resume_from = os.path.join(output_dir, latest)
            print(f'resume: {resume_from}')
    trainer.train(resume_from_checkpoint=resume_from)
    if use_best_checkpoint:
        best_epoch = getattr(trainer.state, 'best_metric', None)
        print(f'best WER: {best_epoch}')
    if mode == 'lora':
        model.save_pretrained(os.path.join(output_dir, 'lora-adapter'))
    else:
        model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f'saved: {output_dir}')
    return (model, processor)
