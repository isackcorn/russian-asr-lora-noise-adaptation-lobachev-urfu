from typing import Sequence

import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def create_compute_metrics(processor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        pad_token_id = processor.tokenizer.pad_token_id
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        pred_norm = [
            normalizer(p).replace("ё", "е").strip() for p in pred_str
        ]
        label_norm = [
            normalizer(l).replace("ё", "е").strip() for l in label_str
        ]

        valid = [
            (p, l) for p, l in zip(pred_norm, label_norm) if len(l) > 0
        ]
        if not valid:
            return {"wer": 100.0, "cer": 100.0}
        preds, labels = zip(*valid)

        wer = 100 * wer_metric.compute(predictions=preds, references=labels)
        cer = 100 * cer_metric.compute(predictions=preds, references=labels)

        return {"wer": round(wer, 2), "cer": round(cer, 2)}

    return compute_metrics


def normalize_russian_text(text: str) -> str:
    normalizer = BasicTextNormalizer()
    return normalizer(text).replace("ё", "е").strip()


def compute_wer_cer(predictions: list, references: list) -> dict:
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    pred_norm = [normalize_russian_text(p) for p in predictions]
    ref_norm = [normalize_russian_text(r) for r in references]

    valid = [(p, l) for p, l in zip(pred_norm, ref_norm) if len(l) > 0]
    if not valid:
        return {"wer": 100.0, "cer": 100.0}
    preds, labels = zip(*valid)

    wer = 100 * wer_metric.compute(predictions=preds, references=labels)
    cer = 100 * cer_metric.compute(predictions=preds, references=labels)

    return {"wer": round(wer, 2), "cer": round(cer, 2)}


def levenshtein_distance(reference: Sequence, hypothesis: Sequence) -> int:
    n = len(reference)
    m = len(hypothesis)
    if n == 0:
        return m
    if m == 0:
        return n

    previous = list(range(m + 1))
    current = [0] * (m + 1)
    for i, ref_item in enumerate(reference, start=1):
        current[0] = i
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current[j] = min(substitution, insertion, deletion)
        previous, current = current, previous
    return previous[m]


def compute_error_counts(prediction: str, reference: str) -> dict:
    prediction_norm = normalize_russian_text(prediction)
    reference_norm = normalize_russian_text(reference)

    pred_words = prediction_norm.split()
    ref_words = reference_norm.split()
    word_errors = levenshtein_distance(ref_words, pred_words)

    pred_chars = list(prediction_norm)
    ref_chars = list(reference_norm)
    char_errors = levenshtein_distance(ref_chars, pred_chars)

    return {
        "prediction_norm": prediction_norm,
        "reference_norm": reference_norm,
        "word_errors": int(word_errors),
        "ref_words": int(len(ref_words)),
        "char_errors": int(char_errors),
        "ref_chars": int(len(ref_chars)),
    }


def compute_wer_cer_from_counts(rows: list[dict]) -> dict:
    word_errors = sum(int(row["word_errors"]) for row in rows)
    ref_words = sum(int(row["ref_words"]) for row in rows)
    char_errors = sum(int(row["char_errors"]) for row in rows)
    ref_chars = sum(int(row["ref_chars"]) for row in rows)

    wer = 100 * word_errors / ref_words if ref_words > 0 else 100.0
    cer = 100 * char_errors / ref_chars if ref_chars > 0 else 100.0
    return {"wer": round(wer, 2), "cer": round(cer, 2)}
