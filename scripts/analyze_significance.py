import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONDITIONS = ["open_stt_calls", "rudevices_10pct", "cv25_test"]
DEFAULT_COMPARISONS = [
    "lora_noisy_r32:zero_shot",
    "lora_noisy_r32:full_ft_clean",
    "lora_noisy_r32:full_ft_noisy",
    "lora_noisy_r32:gigaam_v2_ctc",
]


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paired bootstrap significance for ASR WER/CER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions-dir", type=Path, default=ROOT / "eval" / "predictions_significance")
    p.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "significance")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument(
        "--comparisons",
        nargs="+",
        default=DEFAULT_COMPARISONS,
        metavar="A:B",
        help="Пары моделей. delta = metric(A) - metric(B); отрицательное значение значит A лучше.",
    )
    p.add_argument("--metric", choices=["wer", "cer"], default="wer")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--skip-missing", action="store_true")
    return p


def parse_comparison(spec: str) -> tuple[str, str]:
    sep = ":" if ":" in spec else ","
    if sep not in spec:
        raise SystemExit(f"[ERROR] Сравнение должно быть A:B или A,B, получено: {spec!r}")
    left, right = spec.split(sep, 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise SystemExit(f"[ERROR] Некорректное сравнение: {spec!r}")
    return left, right


def load_prediction_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def metric_keys(metric: str) -> tuple[str, str]:
    if metric == "wer":
        return "word_errors", "ref_words"
    return "char_errors", "ref_chars"


def corpus_metric(errors: np.ndarray, refs: np.ndarray) -> float:
    denom = refs.sum()
    return 100.0 * errors.sum() / denom if denom > 0 else 100.0


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_p_value(diff_boot: np.ndarray, observed_diff: float) -> float:
    n = len(diff_boot)
    if observed_diff < 0:
        tail = (np.sum(diff_boot >= 0) + 1) / (n + 1)
    elif observed_diff > 0:
        tail = (np.sum(diff_boot <= 0) + 1) / (n + 1)
    else:
        return 1.0
    return float(min(1.0, 2.0 * tail))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        adj = min(1.0, p_values[idx] * factor)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def analyze_condition(
    condition: str,
    comparisons: list[tuple[str, str]],
    predictions_dir: Path,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
    skip_missing: bool,
) -> list[dict]:
    involved = sorted({model for pair in comparisons for model in pair})
    rows_by_model: dict[str, dict[str, dict]] = {}

    for model in involved:
        path = predictions_dir / condition / f"{model}.jsonl"
        if not path.exists():
            message = f"[missing] {condition}/{model}.jsonl"
            if skip_missing:
                print(f"[WARN] {message}, пропускаем связанные сравнения")
                continue
            raise SystemExit(f"[ERROR] {message}")
        rows_by_model[model] = load_prediction_rows(path)

    available_comparisons = [
        (left, right) for left, right in comparisons
        if left in rows_by_model and right in rows_by_model
    ]
    if not available_comparisons:
        return []

    involved_available = sorted({model for pair in available_comparisons for model in pair})
    common_ids = set(rows_by_model[involved_available[0]])
    for model in involved_available[1:]:
        common_ids &= set(rows_by_model[model])
    ids = sorted(common_ids)
    if not ids:
        raise SystemExit(f"[ERROR] Нет общих sample id для условия {condition}")

    err_key, ref_key = metric_keys(metric)
    arrays = {}
    for model in involved_available:
        errors = np.array([int(rows_by_model[model][sid][err_key]) for sid in ids], dtype=np.int64)
        refs = np.array([int(rows_by_model[model][sid][ref_key]) for sid in ids], dtype=np.int64)
        arrays[model] = {"errors": errors, "refs": refs}

    observed = {
        model: corpus_metric(arrays[model]["errors"], arrays[model]["refs"])
        for model in involved_available
    }

    boot = {
        model: np.empty(bootstrap_samples, dtype=np.float64)
        for model in involved_available
    }
    n = len(ids)
    for i in range(bootstrap_samples):
        sample_idx = rng.integers(0, n, size=n)
        for model in involved_available:
            model_arrays = arrays[model]
            boot[model][i] = corpus_metric(
                model_arrays["errors"][sample_idx],
                model_arrays["refs"][sample_idx],
            )

    results = []
    for left, right in available_comparisons:
        diff_boot = boot[left] - boot[right]
        observed_diff = observed[left] - observed[right]
        left_ci = percentile_ci(boot[left])
        right_ci = percentile_ci(boot[right])
        diff_ci = percentile_ci(diff_boot)
        p_value = bootstrap_p_value(diff_boot, observed_diff)

        results.append(
            {
                "condition": condition,
                "metric": metric,
                "model_a": left,
                "model_b": right,
                "num_samples": n,
                "model_a_value": round(float(observed[left]), 4),
                "model_b_value": round(float(observed[right]), 4),
                "delta_a_minus_b": round(float(observed_diff), 4),
                "model_a_ci95": [round(left_ci[0], 4), round(left_ci[1], 4)],
                "model_b_ci95": [round(right_ci[0], 4), round(right_ci[1], 4)],
                "delta_ci95": [round(diff_ci[0], 4), round(diff_ci[1], 4)],
                "p_value": p_value,
            }
        )

    return results


def p_fmt(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def ci_fmt(ci: list[float]) -> str:
    return f"[{ci[0]:.2f}; {ci[1]:.2f}]"


def write_markdown(results: list[dict], output_path: Path, alpha: float):
    metric_label = results[0]["metric"].upper()
    lines = [
        "# OOD Statistical Significance",
        "",
        "Метод: paired bootstrap по utterance. "
        "Delta = metric(model_a) - metric(model_b); отрицательная delta означает, что model_a лучше.",
        "",
        f"Holm-Bonferroni correction применена ко всем {len(results)} проверкам в таблице.",
        "",
        f"| Condition | Comparison | N | {metric_label} A | {metric_label} B | Delta | 95% CI Delta | p | p Holm | Significant |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|:---:|",
    ]
    for row in results:
        significant = "yes" if row["p_value_holm"] <= alpha else "no"
        lines.append(
            "| {condition} | {model_a} - {model_b} | {num_samples} | "
            "{model_a_value:.2f} | {model_b_value:.2f} | {delta_a_minus_b:.2f} | "
            "{delta_ci} | {p_value} | {p_value_holm} | {significant} |".format(
                condition=row["condition"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                num_samples=row["num_samples"],
                model_a_value=row["model_a_value"],
                model_b_value=row["model_b_value"],
                delta_a_minus_b=row["delta_a_minus_b"],
                delta_ci=ci_fmt(row["delta_ci95"]),
                p_value=p_fmt(row["p_value"]),
                p_value_holm=p_fmt(row["p_value_holm"]),
                significant=significant,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = get_parser().parse_args()
    comparisons = [parse_comparison(spec) for spec in args.comparisons]
    rng = np.random.default_rng(args.seed)

    all_results = []
    for condition in args.conditions:
        print(f"Анализ условия: {condition}")
        all_results.extend(
            analyze_condition(
                condition=condition,
                comparisons=comparisons,
                predictions_dir=args.predictions_dir,
                metric=args.metric,
                bootstrap_samples=args.bootstrap_samples,
                rng=rng,
                skip_missing=args.skip_missing,
            )
        )

    if not all_results:
        raise SystemExit("[ERROR] Нет результатов для анализа")

    adjusted = holm_bonferroni([row["p_value"] for row in all_results])
    for row, p_adj in zip(all_results, adjusted):
        row["p_value_holm"] = p_adj
        row["significant_holm"] = p_adj <= args.alpha

    metadata = {
        "metric": args.metric,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "alpha": args.alpha,
        "conditions": args.conditions,
        "comparisons": [f"{a}:{b}" for a, b in comparisons],
        "correction": "Holm-Bonferroni across all reported tests",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"ood_{args.metric}_significance.json"
    md_path = args.output_dir / f"ood_{args.metric}_significance.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": all_results}, f, ensure_ascii=False, indent=2)
    write_markdown(all_results, md_path, alpha=args.alpha)

    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
