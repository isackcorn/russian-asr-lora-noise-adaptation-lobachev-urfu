import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def create_compute_metrics(processor):
    wer_metric = evaluate.load('wer')
    cer_metric = evaluate.load('cer')
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        pad_token_id = processor.tokenizer.pad_token_id
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_norm = [normalizer(p).replace('ё', 'е').strip() for p in pred_str]
        label_norm = [normalizer(l).replace('ё', 'е').strip() for l in label_str]
        if any(len(l) == 0 for l in label_norm):
            raise ValueError('empty reference text')
        preds, labels = pred_norm, label_norm
        wer = 100 * wer_metric.compute(predictions=preds, references=labels)
        cer = 100 * cer_metric.compute(predictions=preds, references=labels)
        return {'wer': round(wer, 2), 'cer': round(cer, 2)}
    return compute_metrics

def normalize_russian_text(text: str) -> str:
    normalizer = BasicTextNormalizer()
    return normalizer(text).replace('ё', 'е').strip()

def compute_wer_cer(predictions: list, references: list) -> dict:
    wer_metric = evaluate.load('wer')
    cer_metric = evaluate.load('cer')
    pred_norm = [normalize_russian_text(p) for p in predictions]
    ref_norm = [normalize_russian_text(r) for r in references]
    if any(len(l) == 0 for l in ref_norm):
        raise ValueError('empty reference text')
    preds, labels = pred_norm, ref_norm
    wer = 100 * wer_metric.compute(predictions=preds, references=labels)
    cer = 100 * cer_metric.compute(predictions=preds, references=labels)
    return {'wer': round(wer, 2), 'cer': round(cer, 2)}
