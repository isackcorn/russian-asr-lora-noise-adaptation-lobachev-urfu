import time
import torch
from tqdm import tqdm
from src.utils.metrics import compute_wer_cer

def evaluate_model(model, processor, dataset, device: str='cuda', dtype=torch.bfloat16, desc: str='', max_new_tokens: int=444, batch_size: int=16) -> dict:
    model.to(device)
    model.eval()
    predictions = []
    references = []
    total_time = 0.0
    n = len(dataset)
    with tqdm(total=n, desc=desc) as pbar:
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = [dataset[i] for i in range(start_idx, end_idx)]
            audios = [ex['audio']['array'] for ex in batch]
            sr = batch[0]['audio']['sampling_rate']
            refs = [ex['sentence'] for ex in batch]
            input_features = processor.feature_extractor(audios, sampling_rate=sr, return_tensors='pt').input_features.to(device, dtype=dtype)
            t0 = time.time()
            with torch.no_grad():
                pred_ids = model.generate(input_features, language='russian', task='transcribe', max_new_tokens=max_new_tokens, max_length=None)
            total_time += time.time() - t0
            preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            predictions.extend(preds)
            references.extend(refs)
            pbar.update(len(batch))
    metrics = compute_wer_cer(predictions, references)
    metrics['total_time'] = round(total_time, 2)
    metrics['avg_time_per_sample'] = round(total_time / n, 4) if n > 0 else 0.0
    metrics['num_samples'] = n
    return metrics
