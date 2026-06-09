import argparse
import gc
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.metrics import compute_error_counts, compute_wer_cer_from_counts


ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAME = "gigaam_v2_ctc"

DEFAULT_OOD_MANIFESTS = {
    "open_stt_calls": ROOT / "data" / "open_stt" / "asr_calls_2_val" / "manifest.jsonl",
    "rudevices_10pct": ROOT / "data" / "rudevices" / "manifest_10pct.jsonl",
    "cv25_test": ROOT / "data" / "cv25" / "test_manifest.jsonl",
}


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GigaAM-v2-CTC OOD predictions JSONL for significance testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "predictions_significance")
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_OOD_MANIFESTS))
    p.add_argument(
        "--condition-manifests",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help="Переопределить или добавить OOD-манифесты.",
    )
    p.add_argument("--batch-size", type=int, default=64)
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


def load_manifest_records(manifest_path: Path, limit: int | None = None) -> list[dict]:
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
                    "audio_path": audio_path,
                    "reference": rec.get("text") or rec.get("sentence") or rec.get("transcription") or "",
                    "duration": rec.get("duration", 0),
                }
            )
    return records


def load_counts_summary(path: Path) -> dict:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return {"num_samples": len(rows), **compute_wer_cer_from_counts(rows)}


def write_predictions(model, records: list[dict], out_path: Path, condition: str, args) -> dict:
    if out_path.exists() and not args.force_rerun:
        print(f"  [skip] {BASELINE_NAME} | {condition}: {out_path}")
        return load_counts_summary(out_path)

    from gigaam.preprocess import SAMPLE_RATE, load_audio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    device = model._device
    dtype = model._dtype
    rows_for_summary = []

    with open(tmp_path, "w", encoding="utf-8") as fout:
        with tqdm(total=len(records), desc=f"{BASELINE_NAME} | {condition}") as pbar:
            for start in range(0, len(records), args.batch_size):
                batch = records[start:start + args.batch_size]

                wavs = []
                lengths = []
                for rec in batch:
                    wav = load_audio(rec["audio_path"], SAMPLE_RATE)
                    wavs.append(wav)
                    lengths.append(wav.shape[-1])

                max_len = max(lengths)
                padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
                for i, wav in enumerate(wavs):
                    padded[i, : wav.shape[-1]] = wav
                padded = padded.to(device).to(dtype)
                lengths_t = torch.tensor(lengths, device=device)

                with torch.inference_mode():
                    encoded, encoded_len = model.forward(padded, lengths_t)
                    decoded = model.decoding.decode(model.head, encoded, encoded_len)

                predictions = [text for (text, _, _) in decoded]
                for rec, pred in zip(batch, predictions):
                    counts = compute_error_counts(prediction=pred, reference=rec["reference"])
                    row = {
                        "id": rec["id"],
                        "reference": rec["reference"],
                        "prediction": pred,
                        "duration": rec["duration"],
                        **counts,
                    }
                    rows_for_summary.append(row)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                pbar.update(len(batch))

    os.replace(tmp_path, out_path)
    metrics = compute_wer_cer_from_counts(rows_for_summary)
    print(
        f"  {BASELINE_NAME} | {condition}: "
        f"WER={metrics['wer']:.2f}% CER={metrics['cer']:.2f}% -> {out_path}"
    )
    return {"num_samples": len(rows_for_summary), **metrics}


def main():
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        print("[WARN] CUDA недоступна, GigaAM-инференс будет медленным")

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

    print("=" * 60)
    print("GigaAM-v2-CTC OOD predictions for significance")
    print("=" * 60)
    print(f"  Conditions: {list(selected_manifests)}")
    print(f"  Output: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")

    records_by_condition = {
        name: load_manifest_records(path, limit=args.test_limit)
        for name, path in selected_manifests.items()
    }
    for name, records in records_by_condition.items():
        print(f"  {name}: {len(records)} примеров")

    print("\nЗагрузка GigaAM-v2-CTC...")
    import gigaam

    model = gigaam.load_model("v2_ctc")
    model.eval()
    print(f"  Устройство: {model._device}, dtype: {model._dtype}")

    summary = {}
    for condition, records in records_by_condition.items():
        out_path = args.output_dir / condition / f"{BASELINE_NAME}.jsonl"
        summary[condition] = write_predictions(model, records, out_path, condition, args)

    summary_path = args.output_dir / "summary_gigaam.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({BASELINE_NAME: summary}, f, ensure_ascii=False, indent=2)
    print(f"\nСводка сохранена: {summary_path}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
