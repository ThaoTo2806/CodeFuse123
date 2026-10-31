# Kaggle: đánh giá test cuối cho CE `both`

## Mục đích và quy tắc

Đánh giá cấu hình đã khóa: Qwen2.5-Coder-1.5B-Instruct, prompt `both` (diff + commit message), CE chuẩn, model revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, split seed 42. Cell **không train**, không dò hyperparameter và không chọn seed bằng test. Nó chấm cả ba adapter CE seed 0/1/2 đã được chọn trước dựa trên validation, rồi báo cáo từng seed và tổng hợp.

Đây là lần đánh giá test chính thức cho cấu hình này. Sau khi chạy, không chỉnh prompt/loss/seed dựa vào kết quả test; mọi thay đổi sau đó cần được xem là thí nghiệm mới và cần validation riêng.

## Chuẩn bị Kaggle

1. Gắn dataset HQCM gốc có `train.json` và `test.json`; cell xác minh train SHA-256 và yêu cầu test có 518 mẫu.
2. Đảm bảo ba adapter đã có trong `/kaggle/working` tại các đường dẫn mặc định bên dưới. Nếu chúng nằm ở Kaggle Input hoặc nơi khác, sửa `ADAPTER_DIRS` cho đúng. Adapter phải thuộc CE `both`, không dùng weighted CE.
3. Bật GPU. Qwen là model public; nếu Kaggle không cho tải model, cần bật Internet hoặc cung cấp model cache phù hợp.
4. Cài package nếu cần rồi restart kernel:

```python
%pip install -q transformers==4.46.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

## Cell

```python
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DATA_DIR = Path("/kaggle/working/CodeFuse123/dataset")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
OUTPUT_DIR = Path("/kaggle/working/hqcm_final_test_both")
EXPECTED_TRAIN_SHA256 = "a6430d65f194f5656b4704848d56c7e924bd99532ec78361b18c387a2f44c543"
ADAPTER_DIRS = {
    0: Path("/kaggle/working/hqcm_c2_input_ablation/c2_both_seed0/adapter"),
    1: Path("/kaggle/working/hqcm_c5_seed_confirmation/c5_both_seed1/adapter"),
    2: Path("/kaggle/working/hqcm_c5_seed_confirmation/c5_both_seed2/adapter"),
}
MAX_INPUT_TOKENS = 512
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 51031
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before loading the quantized model.")
if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_DIR}. Set DATA_DIR to the attached HQCM folder."
    )
train_path = DATA_DIR / "train.json"
test_path = DATA_DIR / "test.json"
if not train_path.is_file() or not test_path.is_file():
    raise FileNotFoundError(f"Expected train.json and test.json under {DATA_DIR}.")

import hashlib
train_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
if train_sha256 != EXPECTED_TRAIN_SHA256:
    raise ValueError(
        "The training dataset does not match the C2/C5 dataset. "
        f"Expected SHA-256 {EXPECTED_TRAIN_SHA256}; found {train_sha256}."
    )
with train_path.open(encoding="utf-8") as file:
    train_rows = json.load(file)
with test_path.open(encoding="utf-8") as file:
    test_rows = json.load(file)
if len(train_rows) != 4611 or len(test_rows) != 518:
    raise ValueError(
        f"Expected official split 4611/518; found {len(train_rows)}/{len(test_rows)}."
    )
if any(row.get("type") not in LABELS for row in test_rows):
    raise ValueError("The test set contains labels outside the fixed eight-label set.")
for seed, adapter_dir in ADAPTER_DIRS.items():
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adapter for seed {seed} not found: {adapter_dir}. "
            "Attach the CE-both adapter or set its path in ADAPTER_DIRS."
        )
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights not found under {adapter_dir}.")
    metrics_path = adapter_dir.parent / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Run metrics not found beside seed {seed} adapter: {metrics_path}. "
            "Keep metrics.json beside the adapter so its experiment provenance can be checked."
        )
    with metrics_path.open(encoding="utf-8") as file:
        adapter_metrics = json.load(file)
    if (
        adapter_metrics.get("input_variant") != "both"
        or adapter_metrics.get("seed") != seed
        or adapter_metrics.get("model_revision") != MODEL_REVISION
        or adapter_metrics.get("dataset_train_sha256") != EXPECTED_TRAIN_SHA256
    ):
        raise ValueError(
            f"Adapter provenance mismatch for seed {seed}: {metrics_path}. "
            "Expected input_variant='both', matching seed, pinned revision, and train SHA-256."
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
print("Dataset:", DATA_DIR)
print("Train SHA-256:", train_sha256)
print("Test SHA-256:", test_sha256)
print("Test samples:", len(test_rows))
print("Model revision:", MODEL_REVISION)
print("Evaluation: fixed CE-both adapters, seeds 0/1/2; no training or selection")

# Load the tokenizer once from the same immutable model commit as C2/C5.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=False,
)
tokenizer.truncation_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token_id is None:
    raise ValueError("The tokenizer must define EOS for candidate scoring.")

label_token_ids = {
    label: tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    + [tokenizer.eos_token_id]
    for label in LABELS
}
if any(not token_ids for token_ids in label_token_ids.values()):
    raise ValueError("At least one label has an empty tokenization.")


def build_prompt(row):
    return (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
        f"Diff:\n{row['change']}\n"
        f"Commit message:\n{row['summaries']['en']}\n"
        "Type:"
    )


def score_candidates(model, input_device, prompt):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )["input_ids"]
    sequences = [prompt_ids + label_token_ids[label] for label in LABELS]
    padded_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), padded_length),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        input_ids[row_index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, :len(sequence)] = 1
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        log_probabilities = torch.log_softmax(logits, dim=-1)

    candidate_scores = []
    prompt_length = len(prompt_ids)
    for row_index, label in enumerate(LABELS):
        target_ids = label_token_ids[label]
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(target_ids) - 1,
            device=input_device,
        )
        target_tokens = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=input_device,
        )
        candidate_scores.append(
            log_probabilities[row_index, positions, target_tokens].sum()
        )
    return torch.stack(candidate_scores).cpu()


def calculate_ece(confidence, correct, bins=10):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for bin_index in range(bins):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        in_bin = (
            (confidence >= lower) & (confidence <= upper)
            if bin_index == bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def bootstrap_intervals(gold, predictions, seed):
    rng = np.random.default_rng(seed)
    gold_array = np.asarray(gold)
    prediction_array = np.asarray(predictions)
    accuracy_values = []
    macro_f1_values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(gold_array), size=len(gold_array))
        accuracy_values.append(accuracy_score(gold_array[indices], prediction_array[indices]))
        macro_f1_values.append(f1_score(
            gold_array[indices],
            prediction_array[indices],
            labels=LABELS,
            average="macro",
            zero_division=0,
        ))
    return {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "accuracy_bootstrap_95ci": [
            float(np.quantile(accuracy_values, 0.025)),
            float(np.quantile(accuracy_values, 0.975)),
        ],
        "macro_f1_8_bootstrap_95ci": [
            float(np.quantile(macro_f1_values, 0.025)),
            float(np.quantile(macro_f1_values, 0.975)),
        ],
    }


gold_labels = [str(row["type"]).strip().lower() for row in test_rows]
all_seed_metrics = []

for seed, adapter_dir in ADAPTER_DIRS.items():
    print("\n" + "=" * 72)
    print(f"SCORING LOCKED TEST WITH CE BOTH, SEED {seed}")
    print("Adapter:", adapter_dir)
    print("=" * 72)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    predictions = []
    probabilities = []
    score_rows = []
    started_at = time.perf_counter()
    for index, row in enumerate(test_rows, start=1):
        scores = score_candidates(model, input_device, build_prompt(row))
        class_probabilities = torch.softmax(scores, dim=0).numpy()
        predicted_label = LABELS[int(class_probabilities.argmax())]
        predictions.append(predicted_label)
        probabilities.append(class_probabilities)
        score_rows.append({
            "sample_index": index - 1,
            "gold_label": gold_labels[index - 1],
            "predicted_label": predicted_label,
            "confidence": float(class_probabilities.max()),
            **{
                f"prob_{label}": float(class_probabilities[label_index])
                for label_index, label in enumerate(LABELS)
            },
        })
        if index % 50 == 0 or index == len(test_rows):
            print(f"[seed {seed}] Scored {index}/{len(test_rows)}")
    inference_seconds = time.perf_counter() - started_at

    probability_array = np.asarray(probabilities, dtype=np.float64)
    confidence = probability_array.max(axis=1)
    correct = np.asarray(predictions) == np.asarray(gold_labels)
    report = classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "experiment": "locked_final_test_CE_both",
        "input_variant": "both",
        "loss": "standard_CE",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "dataset_train_sha256": train_sha256,
        "dataset_test_sha256": test_sha256,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "max_input_tokens_left_truncated": MAX_INPUT_TOKENS,
        "inference_seconds": inference_seconds,
        "accuracy": float(accuracy_score(gold_labels, predictions)),
        "macro_f1_8": float(f1_score(
            gold_labels,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )),
        "macro_precision_8": float(precision_score(
            gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )),
        "macro_recall_8": float(recall_score(
            gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )),
        "weighted_f1_8": float(f1_score(
            gold_labels, predictions, labels=LABELS, average="weighted", zero_division=0
        )),
        "ece_10_bins_raw": calculate_ece(confidence, correct),
        "ece_note": "Exploratory, uncalibrated softmax over eight label log-likelihoods.",
        "invalid_count": 0,
        "invalid_note": "Constrained candidate scoring only permits the fixed eight labels.",
        "classification_report": report,
        **bootstrap_intervals(gold_labels, predictions, BOOTSTRAP_SEED + seed),
    }
    metrics_path = OUTPUT_DIR / f"seed{seed}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    pd.DataFrame(score_rows).to_csv(
        OUTPUT_DIR / f"seed{seed}_test_scores.csv",
        index=False,
    )
    pd.DataFrame(confusion_matrix(
        gold_labels,
        predictions,
        labels=LABELS,
    ), index=LABELS, columns=LABELS).to_csv(
        OUTPUT_DIR / f"seed{seed}_confusion_matrix.csv"
    )
    print(classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        digits=4,
    ))
    print(json.dumps({
        key: metrics[key]
        for key in (
            "seed", "accuracy", "macro_f1_8", "macro_precision_8",
            "macro_recall_8", "weighted_f1_8", "ece_10_bins_raw",
            "invalid_count", "inference_seconds",
        )
    }, ensure_ascii=False, indent=2))

    all_seed_metrics.append(metrics)
    del model, base_model
    torch.cuda.empty_cache()

summary_rows = [{
    "seed": item["seed"],
    "accuracy": item["accuracy"],
    "macro_f1_8": item["macro_f1_8"],
    "macro_precision_8": item["macro_precision_8"],
    "macro_recall_8": item["macro_recall_8"],
    "weighted_f1_8": item["weighted_f1_8"],
    "ece_10_bins_raw": item["ece_10_bins_raw"],
    "accuracy_bootstrap_95ci": item["accuracy_bootstrap_95ci"],
    "macro_f1_8_bootstrap_95ci": item["macro_f1_8_bootstrap_95ci"],
} for item in all_seed_metrics]
summary_frame = pd.DataFrame(summary_rows)
summary_frame.to_csv(OUTPUT_DIR / "all_seed_test_summary.csv", index=False)

aggregate = {
    "experiment": "locked_final_test_CE_both_all_seeds",
    "test_evaluated_once_for_preselected_seeds": [0, 1, 2],
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "input_variant": "both",
    "loss": "standard_CE",
    "dataset_train_sha256": train_sha256,
    "dataset_test_sha256": test_sha256,
    "test_samples_per_seed": len(test_rows),
    "seed_results": summary_rows,
    "mean_across_seeds": {
        metric: float(summary_frame[metric].mean())
        for metric in (
            "accuracy", "macro_f1_8", "macro_precision_8",
            "macro_recall_8", "weighted_f1_8", "ece_10_bins_raw",
        )
    },
    "sample_sd_across_seeds": {
        metric: float(summary_frame[metric].std(ddof=1))
        for metric in (
            "accuracy", "macro_f1_8", "macro_precision_8",
            "macro_recall_8", "weighted_f1_8", "ece_10_bins_raw",
        )
    },
    "interpretation_note": (
        "Report all three preselected seeds. Do not select or tune a model based on these test results. "
        "Bootstrap intervals are per-seed over test examples; sample SD describes variation across only three seeds."
    ),
}
with (OUTPUT_DIR / "all_seed_test_metrics.json").open("w", encoding="utf-8") as file:
    json.dump(aggregate, file, ensure_ascii=False, indent=2)
print("\nSaved final test artifacts to:", OUTPUT_DIR)
print(summary_frame.to_string(index=False))
print("Mean across seeds:", json.dumps(aggregate["mean_across_seeds"], indent=2))
print("Sample SD across seeds:", json.dumps(aggregate["sample_sd_across_seeds"], indent=2))
```

## Lưu ý báo cáo

- Cell dùng đủ ba seed đã xác định trước bằng validation, nên báo cáo cả ba; không lấy seed có điểm test cao nhất làm kết quả đại diện.
- `invalid_count = 0` theo thiết kế scoring tập đóng, không phải kết quả generation tự do.
- ECE là thô, chưa calibration. Bootstrap CI theo test examples; SD giữa ba seed thể hiện biến thiên seed với số lượng còn ít.
- Lưu `all_seed_test_metrics.json`, `all_seed_test_summary.csv`, từng `seed*_metrics.json`, `seed*_test_scores.csv`, confusion matrices và log. Không chỉnh cấu hình dựa vào test.
