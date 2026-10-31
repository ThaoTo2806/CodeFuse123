# Kaggle C7: đánh giá trên subtype chưa thấy trong train

## Câu hỏi và giới hạn

C7 kiểm tra liệu mô hình CHCL có phân loại được commit thuộc **type–subtype groups không xuất hiện trong train** hay không. Đây là robustness check trong HQCM, không phải cross-project evaluation: dataset không có repository ID. Nó không thay thế đánh giá trên dataset/project độc lập.

Cell tạo một split cố định theo nhóm `(type, subtype)`, giữ các group validation hoàn toàn ngoài train, và chạy `diff_only`/`both` với seed 0/1/2. Nếu bất kỳ type nào không có ít nhất hai subtype groups, cell dừng vì không thể vừa giữ type trong train vừa đánh giá subtype chưa thấy. Không dùng `test.json`.

## Chuẩn bị và chạy

1. Kaggle: attach dataset HQCM gốc tại `/kaggle/working/CodeFuse123/dataset`, bật GPU.
2. Nếu thiếu package, cài rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 peft==0.13.2 bitsandbytes scikit-learn pandas huggingface_hub
```

3. Chạy cell một lần. Nó lặp hai input variant × ba seed; run đã có metrics phù hợp sẽ được bỏ qua khi chạy tiếp.
4. Gửi `all_experiments_summary.csv`, `split_manifest.json` và metrics/predictions để cập nhật nhật ký.

## Cell

```python
from pathlib import Path
import gc
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATA_DIR = Path("/kaggle/working/CodeFuse123/dataset")
OUTPUT_ROOT = Path("/kaggle/working/hqcm_c7_unseen_subtype")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
INPUT_VARIANTS = ["diff_only", "both"]
TRAIN_SEEDS = [0, 1, 2]
MAX_LENGTH = 512
SPLIT_SEED = 42
VALIDATION_GROUP_FRACTION = 0.20
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
BOOTSTRAP_RESAMPLES = 2000
EXPECTED_TRAIN_SHA256 = "a6430d65f194f5656b4704848d56c7e924bd99532ec78361b18c387a2f44c543"
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before running C7.")
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
TRAIN_PATH = DATA_DIR / "train.json"
if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Raw train.json not found: {TRAIN_PATH}")
train_sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
if train_sha256 != EXPECTED_TRAIN_SHA256:
    raise ValueError(
        f"Train SHA-256 mismatch. Expected {EXPECTED_TRAIN_SHA256}; found {train_sha256}."
    )
with TRAIN_PATH.open(encoding="utf-8") as file:
    raw_rows = json.load(file)
if len(raw_rows) != 4611:
    raise ValueError(f"Expected 4,611 training rows; found {len(raw_rows)}.")
if any(row.get("type") not in LABELS for row in raw_rows):
    raise ValueError("Training data contains a type outside the fixed eight labels.")
if any("subtype" not in row for row in raw_rows):
    raise ValueError("Every sample needs a subtype field to build the grouped split.")


def subtype_group_key(row):
    subtype = row.get("subtype")
    if subtype is None or str(subtype).strip() == "":
        subtype = "__missing_subtype__"
    return (str(row["type"]).strip().lower(), str(subtype).strip())


# Hold out whole within-type subtype groups; keep every top-level type in both splits.
groups_by_type = {label: {} for label in LABELS}
for row_index, row in enumerate(raw_rows):
    label = str(row["type"]).strip().lower()
    group_key = subtype_group_key(row)
    groups_by_type[label].setdefault(group_key, []).append(row_index)

insufficient_groups = {
    label: len(group_map)
    for label, group_map in groups_by_type.items()
    if len(group_map) < 2
}
if insufficient_groups:
    raise RuntimeError(
        "Subtype-held-out validation is not possible while retaining each type in train. "
        f"Need at least two (type, subtype) groups per type; found: {insufficient_groups}"
    )

rng = np.random.default_rng(SPLIT_SEED)
heldout_groups = set()
group_summary = {}
for label_index, label in enumerate(LABELS):
    group_keys = sorted(groups_by_type[label])
    local_rng = np.random.default_rng(SPLIT_SEED + label_index)
    shuffled_keys = [group_keys[index] for index in local_rng.permutation(len(group_keys))]
    holdout_count = max(1, int(round(len(group_keys) * VALIDATION_GROUP_FRACTION)))
    holdout_count = min(holdout_count, len(group_keys) - 1)
    selected_groups = shuffled_keys[:holdout_count]
    heldout_groups.update(selected_groups)
    group_summary[label] = {
        "total_subtype_groups": len(group_keys),
        "heldout_subtype_groups": [group[1] for group in selected_groups],
        "heldout_group_count": len(selected_groups),
    }

validation_indices = [
    index for index, row in enumerate(raw_rows)
    if subtype_group_key(row) in heldout_groups
]
train_indices = [
    index for index, row in enumerate(raw_rows)
    if subtype_group_key(row) not in heldout_groups
]
train_rows = [raw_rows[index] for index in train_indices]
validation_rows = [raw_rows[index] for index in validation_indices]

train_types = {str(row["type"]).strip().lower() for row in train_rows}
validation_types = {str(row["type"]).strip().lower() for row in validation_rows}
if train_types != set(LABELS) or validation_types != set(LABELS):
    raise RuntimeError(
        f"Split must contain all labels. Train={sorted(train_types)}, "
        f"validation={sorted(validation_types)}"
    )
if set(map(subtype_group_key, train_rows)) & set(map(subtype_group_key, validation_rows)):
    raise RuntimeError("Subtype group leakage detected between train and validation.")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
split_manifest = {
    "experiment": "C7_unseen_subtype_generalization",
    "dataset_train_sha256": train_sha256,
    "split_seed": SPLIT_SEED,
    "validation_group_fraction_target": VALIDATION_GROUP_FRACTION,
    "train_samples": len(train_rows),
    "validation_samples": len(validation_rows),
    "train_indices": train_indices,
    "validation_indices": validation_indices,
    "groups_by_type": group_summary,
    "train_counts_by_type": {
        label: sum(row["type"] == label for row in train_rows) for label in LABELS
    },
    "validation_counts_by_type": {
        label: sum(row["type"] == label for row in validation_rows) for label in LABELS
    },
    "validation_subtypes_disjoint_from_train": True,
    "test_json_used": False,
}
with (OUTPUT_ROOT / "split_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(split_manifest, file, ensure_ascii=False, indent=2)

print("Dataset:", DATA_DIR)
print("Train SHA-256:", train_sha256)
print(f"Train/validation samples: {len(train_rows)}/{len(validation_rows)}")
print("Train label counts:", split_manifest["train_counts_by_type"])
print("Validation label counts:", split_manifest["validation_counts_by_type"])
print("Held-out subtype groups by type:", group_summary)

print("Resolving immutable model revision...")
MODEL_REVISION = HfApi().model_info(MODEL_ID, revision="main").sha
if not MODEL_REVISION:
    raise RuntimeError("Could not resolve the Qwen model revision.")
print("Model revision:", MODEL_REVISION)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=False,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def build_prompt(row, variant):
    instruction = (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
    )
    if variant == "diff_only":
        return instruction + f"Diff:\n{row['change']}\nType:"
    return (
        instruction
        + f"Diff:\n{row['change']}\n"
        + f"Commit message:\n{row['summaries']['en']}\nType:"
    )


def encode_row(row, variant):
    prompt_ids = tokenizer(
        build_prompt(row, variant),
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    answer_ids = tokenizer(
        " " + row["type"],
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    answer_ids = answer_ids[:MAX_LENGTH]
    prompt_budget = MAX_LENGTH - len(answer_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def calculate_ece(confidence, correct, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for bin_index in range(bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        in_bin = (
            (confidence >= lower) & (confidence <= upper)
            if bin_index == bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        if in_bin.any():
            value += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(value)


def bootstrap_metrics(gold, predictions, seed):
    rng = np.random.default_rng(seed)
    gold_array = np.asarray(gold)
    prediction_array = np.asarray(predictions)
    accuracy_values = []
    macro_f1_values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(gold_array), size=len(gold_array))
        accuracy_values.append(accuracy_score(gold_array[indices], prediction_array[indices]))
        macro_f1_values.append(f1_score(
            gold_array[indices], prediction_array[indices],
            labels=LABELS, average="macro", zero_division=0,
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


def score_candidates(model, input_device, prompt, candidate_token_ids):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]
    sequences = [prompt_ids + candidate_token_ids[label] for label in LABELS]
    padded_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), padded_length), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        input_ids[row_index, :len(sequence)] = torch.tensor(sequence)
        attention_mask[row_index, :len(sequence)] = 1
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    scores = []
    prompt_length = len(prompt_ids)
    for row_index, label in enumerate(LABELS):
        target_ids = candidate_token_ids[label]
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(target_ids) - 1,
            device=input_device,
        )
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=input_device)
        scores.append(log_probs[row_index, positions, target_tensor].sum())
    return torch.stack(scores).cpu()


candidate_token_ids = {
    label: tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
    for label in LABELS
}
if any(not token_ids for token_ids in candidate_token_ids.values()):
    raise ValueError("A class label tokenized to an empty target.")

base_collator = DataCollatorForSeq2Seq(
    tokenizer,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
    return_tensors="pt",
)


def run_one(input_variant, train_seed):
    run_name = f"c7_unseen_subtype_{input_variant}_seed{train_seed}"
    output_dir = OUTPUT_ROOT / run_name
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (
                previous.get("experiment") == "C7_unseen_subtype_generalization"
                and previous.get("input_variant") == input_variant
                and previous.get("seed") == train_seed
                and previous.get("model_revision") == MODEL_REVISION
                and previous.get("dataset_train_sha256") == train_sha256
            ):
                print(f"Already complete, skipping: {run_name}")
                return previous
        except (OSError, ValueError, TypeError):
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

    raw_train_dataset = Dataset.from_list(train_rows)
    encoded_train = raw_train_dataset.map(
        lambda row: encode_row(row, input_variant),
        remove_columns=raw_train_dataset.column_names,
        desc=f"Tokenizing C7 {input_variant} seed {train_seed}",
    )
    encoded_train = encoded_train.filter(
        lambda row: any(value != -100 for value in row["labels"]),
        desc="Checking non-empty target labels",
    )
    model = None
    trainer = None
    try:
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
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model)
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(base_model, lora_config)
        training_args = TrainingArguments(
            output_dir=str(output_dir / "trainer"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            bf16=False,
            logging_steps=25,
            save_strategy="epoch",
            save_total_limit=1,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
            seed=train_seed,
            data_seed=SPLIT_SEED,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train,
            data_collator=base_collator,
        )
        torch.cuda.reset_peak_memory_stats()
        train_started = time.perf_counter()
        train_result = trainer.train()
        train_seconds = time.perf_counter() - train_started
        peak_memory_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        adapter_dir = output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        model.config.use_cache = True
        model.eval()
        input_device = model.get_input_embeddings().weight.device

        gold_labels = [str(row["type"]).strip().lower() for row in validation_rows]
        predictions = []
        probabilities = []
        for sample_index, row in enumerate(validation_rows, start=1):
            prompt = build_prompt(row, input_variant)
            scores = score_candidates(model, input_device, prompt, candidate_token_ids)
            class_probabilities = torch.softmax(scores, dim=0).numpy()
            probabilities.append(class_probabilities)
            predictions.append(LABELS[int(class_probabilities.argmax())])
            if sample_index % 50 == 0 or sample_index == len(validation_rows):
                print(f"[{run_name}] Scored {sample_index}/{len(validation_rows)}")

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
            "experiment": "C7_unseen_subtype_generalization",
            "run_name": run_name,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "input_variant": input_variant,
            "loss": "standard CE",
            "seed": train_seed,
            "split_seed": SPLIT_SEED,
            "split_type": "group holdout by (type, subtype)",
            "dataset_train_sha256": train_sha256,
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "max_length": MAX_LENGTH,
            "num_train_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_loss": float(train_result.training_loss),
            "train_seconds": train_seconds,
            "peak_gpu_memory_reserved_gb": peak_memory_gb,
            "class_weights_not_used": True,
            "accuracy": float(accuracy_score(gold_labels, predictions)),
            "macro_f1_8": float(f1_score(
                gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
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
            "classification_report": report,
            "heldout_groups_by_type": group_summary,
            **bootstrap_metrics(gold_labels, predictions, 75000 + train_seed),
        }
        prediction_frame = pd.DataFrame({
            "sample_id": [f"train:{index}" for index in validation_indices],
            "gold_label": gold_labels,
            "predicted_label": predictions,
            "confidence": confidence,
        })
        prediction_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
            json.dump({
                **metrics,
                "dtype": "float16",
                "quantization": "4-bit NF4 with double quantization",
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "validation_group_fraction_target": VALIDATION_GROUP_FRACTION,
                "test_json_used": False,
            }, file, ensure_ascii=False, indent=2)
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
        return metrics
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


all_metrics = []
for input_variant in INPUT_VARIANTS:
    for train_seed in TRAIN_SEEDS:
        print("\n" + "=" * 72)
        print(f"STARTING C7 {input_variant}, seed {train_seed}")
        print("=" * 72)
        run_metrics = run_one(input_variant, train_seed)
        all_metrics.append(run_metrics)
        summary_rows = [{
            "run_name": item["run_name"],
            "input_variant": item["input_variant"],
            "seed": item["seed"],
            "accuracy": item["accuracy"],
            "macro_f1_8": item["macro_f1_8"],
            "fix_f1": item["classification_report"]["fix"]["f1-score"],
            "refactor_f1": item["classification_report"]["refactor"]["f1-score"],
            "ece_10_bins_raw": item["ece_10_bins_raw"],
            "train_seconds": item["train_seconds"],
            "peak_gpu_memory_reserved_gb": item["peak_gpu_memory_reserved_gb"],
        } for item in all_metrics]
        pd.DataFrame(summary_rows).to_csv(
            OUTPUT_ROOT / "all_experiments_summary.csv",
            index=False,
        )

print("\nSaved C7 artifacts to:", OUTPUT_ROOT)
print(pd.read_csv(OUTPUT_ROOT / "all_experiments_summary.csv").to_string(index=False))
```

## Cổng đọc kết quả

- So sánh `both` với `diff_only` qua ba seed trên cùng validation groups; báo cáo Macro-F1, Accuracy, F1 từng lớp và CI từng run.
- So sánh với random stratified validation C2/C5 chỉ để lượng hóa mức giảm khi subtype chưa thấy; không gọi đó là benchmark công bằng hoàn toàn vì split protocol khác nhau.
- Kết quả tốt trên C7 hỗ trợ robustness tới subtype groups chưa thấy trong HQCM; không chứng minh cross-project generalization.
- Nếu một type không có tối thiểu hai subtype groups, cell dừng và nêu type đó; không đổi tiêu chí sau khi xem metrics.
