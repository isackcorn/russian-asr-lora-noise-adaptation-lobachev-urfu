import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from datasets import Audio, Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.lora_setup import load_processor
from src.utils.metrics import compute_error_counts, compute_wer_cer_from_counts


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OOD_MANIFESTS = {
    "open_stt_calls": ROOT / "data" / "open_stt" / "asr_calls_2_val" / "manifest.jsonl",
    "rudevices_10pct": ROOT / "data" / "rudevices" / "manifest_10pct.jsonl",
    "cv25_test": ROOT / "data" / "cv25" / "test_manifest.jsonl",
}

ALL_BASELINES = [
    "zero_shot",
    "full_ft_clean",
    "lora_clean",
    "full_ft_noisy",
    "lora_noisy",
    "lora_noisy_r32",
]

DEFAULT_BASELINES = ["zero_shot", "full_ft_clean", "full_ft_noisy", "lora_noisy_r32"]


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whisper OOD predictions JSONL for significance testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints-root", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "predictions_significance")
    p.add_argument("--model-id", type=str, default="openai/whisper-large-v3")
    p.add_argument(
        "--processor-path",
        type=Path,
        default=ROOT / "checkpoints" / "full_ft_clean",
        help="Локальный путь с WhisperProcessor. По умолчанию берём из чекпоинта, чтобы не зависеть от HF Hub.",
    )
    p.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Разрешить обращения к HF Hub для базовой модели. По умолчанию используется локальный cache.",
    )
    p.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES, choices=ALL_BASELINES)
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_OOD_MANIFESTS))
    p.add_argument(
        "--condition-manifests",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help="Переопределить или добавить OOD-манифесты. Пример: cv25_test=data/cv25/test_manifest.jsonl",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=444)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    return p


def parse_condition_manifests(specs: list[str] | None) -> dict[str, Path]:
    manifests = dict(DEFAULT_OOD_MANIFESTS)
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"[ERROR] Ожидается формат NAME=PATH, получено: {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise SystemExit(f"[ERROR] Пустое имя или путь в {spec!r}")
        manifests[name] = Path(raw_path).expanduser().resolve()
    return manifests


def resolve_audio_path(manifest_path: Path, record: dict) -> str:
    raw_path = record.get("wav") or record.get("audio_filepath") or record.get("audio")
    if not raw_path:
        raise KeyError("Запись манифеста должна содержать wav/audio_filepath/audio")
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)

    manifest_dir = manifest_path.resolve().parent
    candidates = [manifest_dir / path, manifest_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((manifest_dir / path).resolve())


def load_manifest_with_ids(manifest_path: Path, limit: int | None = None) -> Dataset:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            audio_path = resolve_audio_path(manifest_path, rec)
            records.append(
                {
                    "id": rec.get("id") or Path(audio_path).stem,
                    "audio": audio_path,
                    "sentence": rec.get("text") or rec.get("sentence") or rec.get("transcription") or "",
                    "duration": rec.get("duration", 0),
                }
            )
    ds = Dataset.from_list(records)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def load_model_for_baseline(baseline: str, args, dtype):
    local_files_only = not args.allow_downloads
    if baseline == "zero_shot":
        print(f"  Загрузка базовой модели: {args.model_id}")
        return WhisperForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

    ckpt_dir = args.checkpoints_root / baseline
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Не найден чекпоинт: {ckpt_dir}")

    if baseline.startswith("full_ft"):
        print(f"  Загрузка Full FT чекпоинта: {ckpt_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=dtype)
        model.config.use_cache = True
        return model

    if baseline.startswith("lora"):
        adapter_dir = ckpt_dir / "lora-adapter"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Не найден LoRA-адаптер: {adapter_dir}")
        print(f"  Загрузка базы: {args.model_id}")
        base = WhisperForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        print(f"  Применение LoRA-адаптера: {adapter_dir}")
        return PeftModel.from_pretrained(base, adapter_dir)

    raise ValueError(f"Неизвестный бейзлайн: {baseline}")


def write_predictions(model, processor, dataset, out_path: Path, label: str, args, device, dtype) -> dict:
    if out_path.exists() and not args.force_rerun:
        print(f"  [skip] {label}: {out_path}")
        return load_counts_summary(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    model.to(device)
    model.eval()

    rows_for_summary = []
    n = len(dataset)
    with open(tmp_path, "w", encoding="utf-8") as fout:
        with tqdm(total=n, desc=label) as pbar:
            for start_idx in range(0, n, args.batch_size):
                end_idx = min(start_idx + args.batch_size, n)
                batch = [dataset[i] for i in range(start_idx, end_idx)]

                audios = [ex["audio"]["array"] for ex in batch]
                sampling_rate = batch[0]["audio"]["sampling_rate"]
                refs = [ex["sentence"] for ex in batch]
                ids = [ex["id"] for ex in batch]
                durations = [ex.get("duration", 0) for ex in batch]

                input_features = processor.feature_extractor(
                    audios, sampling_rate=sampling_rate, return_tensors="pt"
                ).input_features.to(device, dtype=dtype)

                with torch.no_grad():
                    pred_ids = model.generate(
                        input_features,
                        language="russian",
                        task="transcribe",
                        max_new_tokens=args.max_new_tokens,
                        max_length=None,
                    )
                preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

                for sample_id, ref, pred, duration in zip(ids, refs, preds, durations):
                    counts = compute_error_counts(prediction=pred, reference=ref)
                    row = {
                        "id": sample_id,
                        "reference": ref,
                        "prediction": pred,
                        "duration": duration,
                        **counts,
                    }
                    rows_for_summary.append(row)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                pbar.update(len(batch))

    os.replace(tmp_path, out_path)
    metrics = compute_wer_cer_from_counts(rows_for_summary)
    print(f"  {label}: WER={metrics['wer']:.2f}% CER={metrics['cer']:.2f}% -> {out_path}")
    return {"num_samples": len(rows_for_summary), **metrics}


def load_counts_summary(path: Path) -> dict:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return {"num_samples": len(rows), **compute_wer_cer_from_counts(rows)}


def main():
    args = get_parser().parse_args()
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*attention mask.*")
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Whisper large-v3 evaluation требует GPU.")

    manifests = parse_condition_manifests(args.condition_manifests)
    selected_manifests = {}
    for name in args.conditions:
        if name not in manifests:
            raise SystemExit(f"[ERROR] Неизвестное условие {name!r}; задайте --condition-manifests {name}=PATH")
        path = manifests[name]
        if not path.exists():
            raise SystemExit(f"[ERROR] Манифест не найден для {name}: {path}")
        selected_manifests[name] = path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 60)
    print("Whisper OOD predictions for significance")
    print("=" * 60)
    print(f"  Baselines: {args.baselines}")
    print(f"  Conditions: {list(selected_manifests)}")
    print(f"  Output: {args.output_dir}")
    print(f"  Processor: {args.processor_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Local HF cache only: {not args.allow_downloads}")

    print("\nЗагрузка OOD-наборов...")
    datasets = {
        name: load_manifest_with_ids(path, limit=args.test_limit)
        for name, path in selected_manifests.items()
    }
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} примеров")

    if not args.processor_path.exists():
        raise SystemExit(f"[ERROR] Локальный processor не найден: {args.processor_path}")
    processor = load_processor(model_path=str(args.processor_path))
    summary = {}

    for baseline in args.baselines:
        print(f"\n{'=' * 60}\nБейзлайн: {baseline}\n{'=' * 60}")
        model = load_model_for_baseline(baseline, args, dtype)
        summary[baseline] = {}

        for condition, ds in datasets.items():
            out_path = args.output_dir / condition / f"{baseline}.jsonl"
            summary[baseline][condition] = write_predictions(
                model=model,
                processor=processor,
                dataset=ds,
                out_path=out_path,
                label=f"{baseline} | {condition}",
                args=args,
                device=device,
                dtype=dtype,
            )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    summary_path = args.output_dir / "summary_whisper.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nСводка сохранена: {summary_path}")


if __name__ == "__main__":
    main()
