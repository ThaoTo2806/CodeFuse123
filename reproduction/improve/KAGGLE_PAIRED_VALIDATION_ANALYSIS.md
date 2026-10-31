# Paired validation analysis cho C2/C5/C6

## Mục tiêu

Cell này phân tích paired trên validation predictions đã lưu; **không train và không đọc test**. Với từng model và train seed, so sánh `both` với `diff_only` trên đúng cùng sample IDs. Báo cáo Δ Accuracy, Δ Macro-F1, paired bootstrap CI 95%, McNemar exact test cho correctness và Δ F1 `fix`/`refactor`.

Đây là phân tích bổ sung trên validation, không phải một lần đánh giá độc lập mới. Bootstrap CI phản ánh lấy mẫu validation có điều kiện trên các model đã train; ba seed vẫn là số lượng nhỏ. McNemar per seed cũng không thay thế validation ngoài HQCM/project.

## Chuẩn bị Kaggle

1. Attach các artifacts C2, C5 và C6 thành Kaggle Input, hoặc đặt chúng ở các thư mục trong `SEARCH_ROOTS`.
2. Cell tự tìm `metrics.json` và `validation_predictions.csv`. Nếu phát hiện hai artifacts cho cùng model/variant/seed, cell dừng và liệt kê các tệp trùng để tránh chọn nhầm bản.
3. Nếu thiếu package:

```python
%pip install -q pandas numpy scikit-learn scipy
```

4. Chạy cell một lần. Kết quả được lưu trong `/kaggle/working/hqcm_paired_validation_analysis/`.

## Cell

```python
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

SEARCH_ROOTS = [
    Path("/kaggle/input"),
    Path("/kaggle/working/CodeFuse123/reproduction/improve"),
    Path("/kaggle/working/hqcm_c2_input_ablation"),
    Path("/kaggle/working/hqcm_c5_seed_confirmation"),
    Path("/kaggle/working/hqcm_c6_encoder_baselines"),
]
OUTPUT_DIR = Path("/kaggle/working/hqcm_paired_validation_analysis")
EXPECTED_TRAIN_SHA256 = "a6430d65f194f5656b4704848d56c7e924bd99532ec78361b18c387a2f44c543"
EXPECTED_VALIDATION_SAMPLES = 462
EXPECTED_SPLIT_SEED = 42
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 74031
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]


def identify_model(metrics):
    model_id = metrics.get("model_id", "")
    model_key = metrics.get("model_key")
    if model_key:
        return model_key
    if "Qwen2.5-Coder-1.5B-Instruct" in model_id:
        return "qwen"
    return model_id or "unknown"


def experiment_is_in_scope(metrics):
    experiment = metrics.get("experiment", "")
    return experiment in {
        "C2_input_ablation",
        "C5_seed_confirmation",
        "C6_encoder_baseline",
    }



def read_run(metrics_path):
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if not experiment_is_in_scope(metrics):
        return None
    if metrics.get("dataset_train_sha256") != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Unexpected training data hash: {metrics_path}")
    if metrics.get("split_seed") != EXPECTED_SPLIT_SEED:
        raise ValueError(f"Unexpected split seed: {metrics_path}")
    if metrics.get("validation_samples") != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(f"Unexpected validation size: {metrics_path}")
    predictions_path = metrics_path.parent / "validation_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing paired predictions: {predictions_path}")
    frame = pd.read_csv(predictions_path)
    required_columns = {"sample_id", "gold_label", "predicted_label"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns in {predictions_path}")
    if len(frame) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(f"Wrong row count in {predictions_path}: {len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {predictions_path}")
    if not set(frame["gold_label"]).issubset(LABELS):
        raise ValueError(f"Unknown gold labels in {predictions_path}")
    if not set(frame["predicted_label"]).issubset(LABELS):
        raise ValueError(f"Unknown predicted labels in {predictions_path}")

    variant = metrics.get("input_variant", metrics.get("prompt_variant"))
    if variant == "compact":
        variant = "both"
    if variant not in {"both", "diff_only"}:
        return None
    return {
        "model": identify_model(metrics),
        "variant": variant,
        "seed": int(metrics["seed"]),
        "model_revision": metrics.get("model_revision"),
        "frame": frame,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
    }


metrics_files = set()
for root in SEARCH_ROOTS:
    if root.is_dir():
        metrics_files.update(root.rglob("metrics.json"))

runs = {}
duplicates = []
for metrics_path in sorted(metrics_files):
    run = read_run(metrics_path)
    if run is None:
        continue
    key = (run["model"], run["variant"], run["seed"])
    if key in runs:
        duplicates.append((key, runs[key]["metrics_path"], metrics_path))
    else:
        runs[key] = run

if duplicates:
    duplicate_lines = [
        f"{key}: {first} || {second}"
        for key, first, second in duplicates
    ]
    raise RuntimeError(
        "Found multiple artifacts for the same model/variant/seed. "
        "Narrow SEARCH_ROOTS to the intended artifact directories:\n"
        + "\n".join(duplicate_lines)
    )

models = sorted({model for model, _, _ in runs})
print("Discovered models:", models)
print("Discovered runs:", len(runs))
for model in models:
    print(model, "both seeds:", sorted(seed for m, variant, seed in runs if m == model and variant == "both"))
    print(model, "diff_only seeds:", sorted(seed for m, variant, seed in runs if m == model and variant == "diff_only"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
paired_rows = []

for model in models:
    seeds = sorted({
        seed for found_model, variant, seed in runs
        if found_model == model and variant in {"both", "diff_only"}
    })
    for seed in seeds:
        both_run = runs.get((model, "both", seed))
        diff_run = runs.get((model, "diff_only", seed))
        if both_run is None or diff_run is None:
            print(f"Skipping incomplete pair: model={model}, seed={seed}")
            continue

        both_frame = both_run["frame"].sort_values("sample_id").reset_index(drop=True)
        diff_frame = diff_run["frame"].sort_values("sample_id").reset_index(drop=True)
        if not both_frame["sample_id"].equals(diff_frame["sample_id"]):
            raise ValueError(f"Paired sample IDs differ for {model}, seed {seed}.")
        if not both_frame["gold_label"].equals(diff_frame["gold_label"]):
            raise ValueError(f"Gold labels differ for {model}, seed {seed}.")
        safe_model = model.replace("/", "_").replace("\\", "_")
        confusion_matrix(
            both_frame["gold_label"],
            both_frame["predicted_label"],
            labels=LABELS,
        ).tofile(OUTPUT_DIR / f"{safe_model}_seed{seed}_both_confusion.csv", sep=",")
        confusion_matrix(
            diff_frame["gold_label"],
            diff_frame["predicted_label"],
            labels=LABELS,
        ).tofile(OUTPUT_DIR / f"{safe_model}_seed{seed}_diff_only_confusion.csv", sep=",")

        gold = both_frame["gold_label"].to_numpy()
        both_pred = both_frame["predicted_label"].to_numpy()
        diff_pred = diff_frame["predicted_label"].to_numpy()
        both_correct = both_pred == gold
        diff_correct = diff_pred == gold

        both_accuracy = accuracy_score(gold, both_pred)
        diff_accuracy = accuracy_score(gold, diff_pred)
        both_macro_f1 = f1_score(
            gold, both_pred, labels=LABELS, average="macro", zero_division=0
        )
        diff_macro_f1 = f1_score(
            gold, diff_pred, labels=LABELS, average="macro", zero_division=0
        )
        both_fix_f1 = f1_score(
            gold, both_pred, labels=LABELS, average=None, zero_division=0
        )[LABELS.index("fix")]
        diff_fix_f1 = f1_score(
            gold, diff_pred, labels=LABELS, average=None, zero_division=0
        )[LABELS.index("fix")]
        both_refactor_f1 = f1_score(
            gold, both_pred, labels=LABELS, average=None, zero_division=0
        )[LABELS.index("refactor")]
        diff_refactor_f1 = f1_score(
            gold, diff_pred, labels=LABELS, average=None, zero_division=0
        )[LABELS.index("refactor")]

        # Paired bootstrap resamples the same validation examples for both systems.
        rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
        delta_accuracy_samples = []
        delta_macro_f1_samples = []
        gold_array = np.asarray(gold)
        both_array = np.asarray(both_pred)
        diff_array = np.asarray(diff_pred)
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample_indices = rng.integers(0, len(gold_array), size=len(gold_array))
            sampled_gold = gold_array[sample_indices]
            sampled_both = both_array[sample_indices]
            sampled_diff = diff_array[sample_indices]
            delta_accuracy_samples.append(
                accuracy_score(sampled_gold, sampled_both)
                - accuracy_score(sampled_gold, sampled_diff)
            )
            delta_macro_f1_samples.append(
                f1_score(
                    sampled_gold, sampled_both, labels=LABELS,
                    average="macro", zero_division=0,
                )
                - f1_score(
                    sampled_gold, sampled_diff, labels=LABELS,
                    average="macro", zero_division=0,
                )
            )

        both_only_correct = int(np.sum(both_correct & ~diff_correct))
        diff_only_correct = int(np.sum(diff_correct & ~both_correct))
        discordant = both_only_correct + diff_only_correct
        mcnemar_p = (
            float(binomtest(
                min(both_only_correct, diff_only_correct),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue)
            if discordant
            else 1.0
        )

        paired_rows.append({
            "model": model,
            "seed": seed,
            "both_revision": both_run["model_revision"],
            "diff_revision": diff_run["model_revision"],
            "both_accuracy": both_accuracy,
            "diff_only_accuracy": diff_accuracy,
            "delta_accuracy_both_minus_diff": both_accuracy - diff_accuracy,
            "delta_accuracy_bootstrap_95ci_low": float(np.quantile(delta_accuracy_samples, 0.025)),
            "delta_accuracy_bootstrap_95ci_high": float(np.quantile(delta_accuracy_samples, 0.975)),
            "both_macro_f1_8": both_macro_f1,
            "diff_only_macro_f1_8": diff_macro_f1,
            "delta_macro_f1_both_minus_diff": both_macro_f1 - diff_macro_f1,
            "delta_macro_f1_bootstrap_95ci_low": float(np.quantile(delta_macro_f1_samples, 0.025)),
            "delta_macro_f1_bootstrap_95ci_high": float(np.quantile(delta_macro_f1_samples, 0.975)),
            "both_fix_f1": both_fix_f1,
            "diff_only_fix_f1": diff_fix_f1,
            "delta_fix_f1": both_fix_f1 - diff_fix_f1,
            "both_refactor_f1": both_refactor_f1,
            "diff_only_refactor_f1": diff_refactor_f1,
            "delta_refactor_f1": both_refactor_f1 - diff_refactor_f1,
            "both_only_correct": both_only_correct,
            "diff_only_correct": diff_only_correct,
            "mcnemar_exact_p": mcnemar_p,
            "n_validation": len(gold),
            "both_metrics_path": str(both_run["metrics_path"]),
            "diff_metrics_path": str(diff_run["metrics_path"]),
        })

if not paired_rows:
    raise RuntimeError(
        "No complete both/diff_only pairs found. Attach C2/C5 Qwen and C6 encoder artifacts."
    )

paired_frame = pd.DataFrame(paired_rows)
paired_frame.to_csv(OUTPUT_DIR / "paired_results_by_seed.csv", index=False)
summary_rows = []
for model, group in paired_frame.groupby("model", sort=True):
    summary_rows.append({
        "model": model,
        "paired_seeds": len(group),
        "both_accuracy_mean": group["both_accuracy"].mean(),
        "diff_only_accuracy_mean": group["diff_only_accuracy"].mean(),
        "delta_accuracy_mean": group["delta_accuracy_both_minus_diff"].mean(),
        "delta_accuracy_seed_sd": group["delta_accuracy_both_minus_diff"].std(ddof=1),
        "both_macro_f1_mean": group["both_macro_f1_8"].mean(),
        "diff_only_macro_f1_mean": group["diff_only_macro_f1_8"].mean(),
        "delta_macro_f1_mean": group["delta_macro_f1_both_minus_diff"].mean(),
        "delta_macro_f1_seed_sd": group["delta_macro_f1_both_minus_diff"].std(ddof=1),
        "delta_fix_f1_mean": group["delta_fix_f1"].mean(),
        "delta_refactor_f1_mean": group["delta_refactor_f1"].mean(),
        "seeds_both_higher_macro_f1": int((group["delta_macro_f1_both_minus_diff"] > 0).sum()),
        "interpretation": "Descriptive only: three seeds, one HQCM validation split.",
    })
summary_frame = pd.DataFrame(summary_rows)
summary_frame.to_csv(OUTPUT_DIR / "paired_results_summary.csv", index=False)

print("\nPaired results by seed:")
print(paired_frame.to_string(index=False))
print("\nPaired summary by model:")
print(summary_frame.to_string(index=False))
print("\nSaved analysis to:", OUTPUT_DIR)
```

## Diễn giải đúng

- `delta_*_both_minus_diff` dùng cùng seed và cùng validation IDs. CI bootstrap ghép cặp lấy lại cùng chỉ số mẫu cho cả hai biến thể.
- `mcnemar_exact_p` kiểm tra bất đồng đúng/sai theo từng seed. Đây là phân tích validation; không dùng p-value để chọn model sau khi đã xem hàng loạt kết quả.
- SD giữa 3 seed là mô tả biến thiên seed, không phải CI. Không gộp 3 seed thành 1.386 quan sát độc lập.
- ECE là thô, chưa calibration. Cell không đọc test.
- Kết quả vẫn thuộc một split của HQCM. Muốn tuyên bố cross-project/generalization cần dataset hoặc project split có provenance.
