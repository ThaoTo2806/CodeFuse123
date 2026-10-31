# Kaggle C6: baseline encoder cùng protocol validation

## Vì sao chạy cell này

Khoảng trống lớn nhất hiện là thiếu baseline encoder trên cùng validation split. Cell này so sánh **CodeBERT** và **GraphCodeBERT** với đầu vào `diff-only` và `diff + summary`, mỗi cấu hình qua train seed 0/1/2: tổng cộng 12 runs trong một vòng lặp. Kết quả được so với Qwen C2/C5 trên cùng 4.149/462 split và cùng 8 nhãn.

Cell chỉ dùng validation, không đọc `test.json`. Revision SHA của từng encoder được lấy qua Hugging Face Hub rồi pin cho tokenizer/model. Sau khi hoàn tất, cell lưu metrics, report theo lớp, bootstrap CI và validation predictions; run đã có metrics đúng cấu hình sẽ tự bỏ qua khi chạy tiếp.

Đây là baseline encoder trên một split HQCM, không phải đánh giá cross-project. Raw dataset hiện không có repository/project ID; để chứng minh tổng quát hóa qua project/dataset cần thêm dữ liệu có provenance phù hợp.

## Chuẩn bị Kaggle

1. Gắn dataset HQCM gốc tại `/kaggle/working/CodeFuse123/dataset` hoặc sửa `DATA_DIR`.
2. Bật GPU và Internet để tải hai encoder từ Hugging Face.
3. Nếu thiếu package, cài rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 huggingface_hub scikit-learn pandas
```

4. Chạy toàn bộ cell một lần. Nó lặp model × input variant × seed; không cần đổi tiêu đề hay chạy cell riêng cho từng cấu hình. Có thể chạy lại sau khi gián đoạn: mỗi run có metrics và predictions lưu ngay sau khi hoàn tất.
5. Tải `/kaggle/working/hqcm_c6_encoder_baselines/` về và gửi `all_experiments_summary.csv` cùng metrics khi xong.

## Cell

```python
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATA_DIR = Path("/kaggle/working/CodeFuse123/dataset")
OUTPUT_ROOT = Path("/kaggle/working/hqcm_c6_encoder_baselines")
MODEL_SPECS = {
    "codebert": "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
}
INPUT_VARIANTS = ["diff_only", "both"]
TRAIN_SEEDS = [0, 1, 2]
MAX_LENGTH = 512
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 42
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
BOOTSTRAP_RESAMPLES = 2000
EXPECTED_TRAIN_SHA256 = "a6430d65f194f5656b4704848d56c7e924bd99532ec78361b18c387a2f44c543"
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before running the encoder baseline cell.")
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

TRAIN_PATH = DATA_DIR / "train.json"
if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Raw train.json not found: {TRAIN_PATH}")
train_sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
if train_sha256 != EXPECTED_TRAIN_SHA256:
    raise ValueError(
        "The training dataset does not match C2/C5. "
        f"Expected {EXPECTED_TRAIN_SHA256}; found {train_sha256}."
    )
with TRAIN_PATH.open(encoding="utf-8") as file:
    raw_rows = json.load(file)
if len(raw_rows) != 4611:
    raise ValueError(f"Expected 4,611 training rows; found {len(raw_rows)}.")
if any(row.get("type") not in LABELS for row in raw_rows):
    raise ValueError("The training data contains labels outside the fixed eight-label set.")
if any(not row.get("change") for row in raw_rows):
    raise ValueError("At least one sample has a missing/empty change diff.")
if any(not row.get("summaries", {}).get("en") for row in raw_rows):
    raise ValueError("At least one sample has a missing summaries.en value.")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
train_indices, validation_indices = train_test_split(
    np.arange(len(raw_rows)),
    test_size=VALIDATION_FRACTION,
    random_state=SPLIT_SEED,
    stratify=[row["type"] for row in raw_rows],
)
train_rows = [raw_rows[int(index)] for index in train_indices]
validation_rows = [raw_rows[int(index)] for index in validation_indices]
with (OUTPUT_ROOT / "split_indices.json").open("w", encoding="utf-8") as file:
    json.dump({
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "train_indices": [int(index) for index in train_indices],
        "validation_indices": [int(index) for index in validation_indices],
    }, file, ensure_ascii=False, indent=2)

print("Dataset:", DATA_DIR)
print("Train SHA-256:", train_sha256)
print(f"Train/validation: {len(train_rows)}/{len(validation_rows)}")
print("Models:", MODEL_SPECS)
print("Variants:", INPUT_VARIANTS)
print("Seeds:", TRAIN_SEEDS)
print("GPU:", torch.cuda.get_device_name(0))


def calculate_ece(confidence, correct, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for bin_index in range(bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
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
        sample_indices = rng.integers(0, len(gold_array), size=len(gold_array))
        accuracy_values.append(accuracy_score(
            gold_array[sample_indices], prediction_array[sample_indices]
        ))
        macro_f1_values.append(f1_score(
            gold_array[sample_indices],
            prediction_array[sample_indices],
            labels=list(range(len(LABELS))),
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


def make_tokenizer_function(tokenizer, input_variant):
    def tokenize_batch(batch):
        if input_variant == "diff_only":
            encoded = tokenizer(
                batch["change"],
                truncation=True,
                max_length=MAX_LENGTH,
            )
        else:
            encoded = tokenizer(
                batch["change"],
                batch["summary_en"],
                truncation="only_first",
                max_length=MAX_LENGTH,
            )
        encoded["labels"] = [LABEL_TO_ID[label] for label in batch["type"]]
        return encoded
    return tokenize_batch


all_run_metrics = []
api = HfApi()

for model_key, model_id in MODEL_SPECS.items():
    model_revision = api.model_info(model_id).sha
    if not model_revision:
        raise RuntimeError(f"Could not resolve immutable model revision for {model_id}.")
    print(f"\nPinned {model_key} revision: {model_revision}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        use_fast=True,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    for input_variant in INPUT_VARIANTS:
        run_train_rows = [{
            "change": row["change"],
            "summary_en": row["summaries"]["en"],
            "type": row["type"],
        } for row in train_rows]
        run_validation_rows = [{
            "change": row["change"],
            "summary_en": row["summaries"]["en"],
            "type": row["type"],
        } for row in validation_rows]
        train_dataset = Dataset.from_list(run_train_rows)
        validation_dataset = Dataset.from_list(run_validation_rows)
        tokenize_batch = make_tokenizer_function(tokenizer, input_variant)
        encoded_train = train_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc=f"Tokenizing {model_key} {input_variant} train",
        )
        encoded_validation = validation_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=validation_dataset.column_names,
            desc=f"Tokenizing {model_key} {input_variant} validation",
        )

        for train_seed in TRAIN_SEEDS:
            run_name = f"c6_{model_key}_{input_variant}_seed{train_seed}"
            output_dir = OUTPUT_ROOT / run_name
            metrics_path = output_dir / "metrics.json"
            if metrics_path.is_file():
                try:
                    previous = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if (
                        previous.get("experiment") == "C6_encoder_baseline"
                        and previous.get("model_revision") == model_revision
                        and previous.get("input_variant") == input_variant
                        and previous.get("seed") == train_seed
                        and previous.get("dataset_train_sha256") == train_sha256
                    ):
                        print(f"Already complete, skipping: {run_name}")
                        all_run_metrics.append(previous)
                        continue
                except (OSError, ValueError, TypeError):
                    pass

            output_dir.mkdir(parents=True, exist_ok=True)
            set_seed(train_seed)
            print("\n" + "=" * 72)
            print(f"STARTING {run_name}")
            print("Model revision:", model_revision)
            print("=" * 72)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                revision=model_revision,
                num_labels=len(LABELS),
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID,
            )
            training_args = TrainingArguments(
                output_dir=str(output_dir / "trainer"),
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                fp16=True,
                bf16=False,
                logging_steps=50,
                save_strategy="no",
                report_to="none",
                seed=train_seed,
                data_seed=SPLIT_SEED,
                dataloader_num_workers=2,
                remove_unused_columns=True,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=encoded_train,
                data_collator=data_collator,
            )
            torch.cuda.reset_peak_memory_stats()
            train_started = time.perf_counter()
            train_result = trainer.train()
            train_seconds = time.perf_counter() - train_started
            peak_memory_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

            prediction_output = trainer.predict(encoded_validation)
            logits = prediction_output.predictions
            probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
            predicted_ids = probabilities.argmax(axis=1)
            predictions = [ID_TO_LABEL[int(label_id)] for label_id in predicted_ids]
            gold_labels = [row["type"] for row in validation_rows]
            confidence = probabilities.max(axis=1)
            correct = np.asarray(predictions) == np.asarray(gold_labels)
            report = classification_report(
                gold_labels,
                predictions,
                labels=LABELS,
                target_names=LABELS,
                output_dict=True,
                zero_division=0,
            )
            bootstrap = bootstrap_intervals(
                [LABEL_TO_ID[label] for label in gold_labels],
                predicted_ids,
                62000 + train_seed,
            )
            metrics = {
                "experiment": "C6_encoder_baseline",
                "run_name": run_name,
                "model_key": model_key,
                "model_id": model_id,
                "model_revision": model_revision,
                "input_variant": input_variant,
                "seed": train_seed,
                "split_seed": SPLIT_SEED,
                "dataset_train_sha256": train_sha256,
                "train_samples": len(train_rows),
                "validation_samples": len(validation_rows),
                "max_length": MAX_LENGTH,
                "truncation": "only_first for paired diff+summary; True for diff-only",
                "num_train_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "train_loss": float(train_result.training_loss),
                "train_seconds": train_seconds,
                "peak_gpu_memory_reserved_gb": peak_memory_gb,
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
                "ece_note": "Exploratory, uncalibrated softmax over encoder logits.",
                "classification_report": report,
                **bootstrap,
            }
            pd.DataFrame({
                "sample_id": [f"train:{int(index)}" for index in validation_indices],
                "gold_label": gold_labels,
                "predicted_label": predictions,
                "confidence": confidence,
            }).to_csv(output_dir / "validation_predictions.csv", index=False)
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(metrics, file, ensure_ascii=False, indent=2)
            with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
                json.dump({
                    **metrics,
                    "dtype": "float16",
                    "loss": "standard cross-entropy",
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                }, file, ensure_ascii=False, indent=2)
            all_run_metrics.append(metrics)
            pd.DataFrame([{
                "run_name": item["run_name"],
                "model_key": item["model_key"],
                "model_revision": item["model_revision"],
                "input_variant": item["input_variant"],
                "seed": item["seed"],
                "accuracy": item["accuracy"],
                "macro_f1_8": item["macro_f1_8"],
                "fix_f1": item["classification_report"]["fix"]["f1-score"],
                "refactor_f1": item["classification_report"]["refactor"]["f1-score"],
                "ece_10_bins_raw": item["ece_10_bins_raw"],
                "train_seconds": item["train_seconds"],
                "peak_gpu_memory_reserved_gb": item["peak_gpu_memory_reserved_gb"],
            } for item in all_run_metrics]).to_csv(
                OUTPUT_ROOT / "all_experiments_summary.csv",
                index=False,
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
                    "run_name", "accuracy", "macro_f1_8", "macro_precision_8",
                    "macro_recall_8", "ece_10_bins_raw", "train_seconds",
                    "peak_gpu_memory_reserved_gb",
                )
            }, ensure_ascii=False, indent=2))
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

print("\nSaved C6 artifacts to:", OUTPUT_ROOT)
summary_path = OUTPUT_ROOT / "all_experiments_summary.csv"
if summary_path.is_file():
    print(pd.read_csv(summary_path).to_string(index=False))
```

## Cách đọc

- So sánh encoder với Qwen `both`/`diff-only` bằng validation Macro-F1, Accuracy, F1 `fix`/`refactor`, ECE thô và chi phí. Qwen validation: `both` Macro-F1 0,8475 ± 0,0149; `diff-only` 0,7927 ± 0,0055.
- So sánh `both` với `diff-only` bên trong từng encoder để xem hiệu quả của message có lặp lại qua kiến trúc khác hay không.
- Encoder nhận diff/summary theo cặp tokenizer (không instruction prompt như Qwen); vì vậy đây là baseline kiến trúc cùng dữ liệu/split, không phải so sánh serialization hoàn toàn đồng nhất.
- Đây là cùng một random stratified split, không phải cross-project evaluation. ECE không hiệu chỉnh; test chính thức không được dùng trong cell.
- Không tuyên bố encoder hay SLM thắng tổng quát chỉ từ một split. Dùng kết quả để xác định ứng viên tiếp theo, không mở lại test để chọn.
