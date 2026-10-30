# Kaggle C4: weighted CE cho lớp commit

## Mục tiêu

Kiểm tra weighted cross-entropy có giúp lớp khó/thiểu số mà không làm giảm kết quả tổng thể hay không. Giữ cố định đầu vào `both` (diff + commit message), split, model, revision và scoring từ C2/C5; chỉ đổi loss.

Cell chạy liên tiếp `TRAIN_SEEDS = [0, 1, 2]` trong một lần gọi. Baseline CE cùng seed đã có từ C2/C5, nên cell chỉ train weighted CE; không cần đổi tiêu đề hay chạy cell riêng cho từng seed. Sau mỗi seed, cell lưu metrics và cập nhật CSV để có thể tiếp tục nếu phiên bị ngắt. Không đọc test chính thức.

## So sánh

| Nhánh       | Nguồn baseline       | Accuracy | Macro-F1 (8 nhãn) |
| :---------- | :------------------- | -------: | ----------------: |
| CE seed 0   | C2 `both` seed 0     |   0,7879 |            0,8314 |
| CE seed 1   | C5 `both` seed 1     |   0,7965 |            0,8503 |
| CE seed 2   | C5 `both` seed 2     |   0,8182 |            0,8607 |
| Weighted CE | Cell này, seed 0/1/2 | Chờ chạy |          Chờ chạy |

Trọng số mỗi lớp là `N / (8 * số_mẫu_lớp)` tính từ phần train của split cố định. Cell áp trọng số của nhãn mẫu lên loss từng token được giám sát, bao gồm EOS. Đây là một pilot về weighted CE; chưa so với focal loss.

## Chuẩn bị

1. Kaggle: attach đúng dataset HQCM gốc có `train.json`, bật GPU.
2. Nếu thiếu package, cài rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

3. Chạy toàn bộ cell một lần. Nó sẽ huấn luyện 3 seed tuần tự, dự kiến vài giờ GPU. Nếu phải tiếp tục sau gián đoạn, chạy lại cell: các seed đã lưu đủ metrics sẽ tự bỏ qua.
4. Tải `/kaggle/working/hqcm_c4_weighted_ce/` về trước khi phiên kết thúc; gửi `all_experiments_summary.csv` cùng metrics/predictions để cập nhật nhật ký.

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
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
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
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
OUTPUT_ROOT = Path("/kaggle/working/hqcm_c4_weighted_ce")
TRAIN_SEEDS = [0, 1, 2]
MAX_LENGTH = 512
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 42
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
BOOTSTRAP_RESAMPLES = 2000
EXPECTED_TRAIN_SHA256 = "a6430d65f194f5656b4704848d56c7e924bd99532ec78361b18c387a2f44c543"
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]
BASELINE_CE = {
    0: {"accuracy": 0.7878787878787878, "macro_f1_8": 0.8314135673632241},
    1: {"accuracy": 0.7965367965367965, "macro_f1_8": 0.8503386807369344},
    2: {"accuracy": 0.8181818181818182, "macro_f1_8": 0.8607361382104403},
}

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before loading the quantized model.")
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

TRAIN_PATH = DATA_DIR / "train.json"
if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Raw train.json not found: {TRAIN_PATH}")
train_file_hash = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
if train_file_hash != EXPECTED_TRAIN_SHA256:
    raise ValueError(
        "The train.json SHA-256 does not match C2/C5. "
        f"Expected {EXPECTED_TRAIN_SHA256}; found {train_file_hash}."
    )
with TRAIN_PATH.open(encoding="utf-8") as file:
    raw_rows = json.load(file)
if len(raw_rows) != 4611:
    raise ValueError(f"Expected 4,611 training rows; found {len(raw_rows)}.")
if any(row.get("type") not in LABELS for row in raw_rows):
    raise ValueError("The dataset contains labels outside the fixed eight-label set.")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
train_indices, validation_indices = train_test_split(
    np.arange(len(raw_rows)),
    test_size=VALIDATION_FRACTION,
    random_state=SPLIT_SEED,
    stratify=[row["type"] for row in raw_rows],
)
train_rows = [raw_rows[int(index)] for index in train_indices]
validation_rows = [raw_rows[int(index)] for index in validation_indices]

train_counts = {label: sum(row["type"] == label for row in train_rows) for label in LABELS}
class_weights = {
    label: len(train_rows) / (len(LABELS) * count)
    for label, count in train_counts.items()
}
print("Train/validation:", len(train_rows), len(validation_rows))
print("Dataset SHA-256:", train_file_hash)
print("Class counts:", train_counts)
print("Normalized inverse-frequency weights:", class_weights)

with (OUTPUT_ROOT / "split_indices.json").open("w", encoding="utf-8") as file:
    json.dump({
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "train_indices": [int(index) for index in train_indices],
        "validation_indices": [int(index) for index in validation_indices],
    }, file, ensure_ascii=False, indent=2)


def build_prompt(row):
    return (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
        f"Diff:\n{row['change']}\n"
        f"Commit message:\n{row['summaries']['en']}\n"
        "Type:"
    )


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=False,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def encode_example(row):
    prompt_ids = tokenizer(
        build_prompt(row),
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
    target_ids = prompt_ids + answer_ids
    return {
        "input_ids": target_ids,
        "attention_mask": [1] * len(target_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "class_weight": class_weights[row["type"]],
    }


train_dataset = Dataset.from_list(train_rows)
encoded_train_dataset = train_dataset.map(
    encode_example,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing weighted-CE training split",
)
encoded_train_dataset = encoded_train_dataset.filter(
    lambda row: any(label != -100 for label in row["labels"]),
    desc="Checking target labels",
)
base_collator = DataCollatorForSeq2Seq(
    tokenizer,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
    return_tensors="pt",
)


class WeightedBatchCollator:
    def __call__(self, features):
        example_weights = torch.tensor(
            [feature["class_weight"] for feature in features],
            dtype=torch.float32,
        )
        model_features = [
            {key: value for key, value in feature.items() if key != "class_weight"}
            for feature in features
        ]
        batch = base_collator(model_features)
        batch["class_weight"] = example_weights
        return batch


class ClassWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch_weights = inputs.pop("class_weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        batch_weights = batch_weights.to(outputs.logits.device)
        shifted_logits = outputs.logits[..., :-1, :].contiguous().float()
        shifted_labels = labels[..., 1:].contiguous()
        batch_size, sequence_length, vocabulary_size = shifted_logits.shape
        token_losses = F.cross_entropy(
            shifted_logits.view(-1, vocabulary_size),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(batch_size, sequence_length)
        valid_tokens = shifted_labels.ne(-100)
        token_weights = batch_weights.unsqueeze(1).expand_as(token_losses)
        weighted_denominator = (token_weights * valid_tokens).sum().clamp_min(1.0)
        loss = (token_losses * token_weights * valid_tokens).sum() / weighted_denominator
        return (loss, outputs) if return_outputs else loss


def expected_calibration_error(confidences, correct_flags, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for bin_index in range(bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        in_bin = (
            (confidences >= lower) & (confidences <= upper)
            if bin_index == bins - 1
            else (confidences >= lower) & (confidences < upper)
        )
        if in_bin.any():
            error += in_bin.mean() * abs(
                correct_flags[in_bin].mean() - confidences[in_bin].mean()
            )
    return float(error)


def score_prompt(model, prompt, label_token_ids, input_device):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]
    sequences = [prompt_ids + label_token_ids[label] for label in LABELS]
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
        log_probabilities = torch.log_softmax(logits, dim=-1)
    scores = []
    prompt_length = len(prompt_ids)
    for row_index, label in enumerate(LABELS):
        target_ids = label_token_ids[label]
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(target_ids) - 1,
            device=input_device,
        )
        targets = torch.tensor(target_ids, dtype=torch.long, device=input_device)
        scores.append(log_probabilities[row_index, positions, targets].sum())
    return torch.stack(scores).cpu()


def run_weighted_seed(train_seed):
    run_name = f"c4_weighted_ce_seed{train_seed}"
    output_dir = OUTPUT_ROOT / run_name
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            previous.get("experiment") == "C4_weighted_CE"
            and previous.get("seed") == train_seed
            and previous.get("model_revision") == MODEL_REVISION
            and previous.get("dataset_train_sha256") == train_file_hash
        ):
            print(f"Already completed: {run_name}; skipping.")
            return previous

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

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
    model.print_trainable_parameters()
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
    trainer = ClassWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,
        data_collator=WeightedBatchCollator(),
    )
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_started
    peak_gpu_memory_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    model.config.use_cache = True
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    label_token_ids = {
        label: tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
        for label in LABELS
    }
    gold_labels = [row["type"] for row in validation_rows]
    predictions = []
    probabilities = []
    prompt_lengths = []
    for index, row in enumerate(validation_rows, start=1):
        prompt = build_prompt(row)
        prompt_lengths.append(len(tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]))
        scores = score_prompt(model, prompt, label_token_ids, input_device)
        class_probabilities = torch.softmax(scores, dim=0).numpy()
        predictions.append(LABELS[int(class_probabilities.argmax())])
        probabilities.append(class_probabilities)
        if index % 50 == 0 or index == len(validation_rows):
            print(f"[{run_name}] Scored {index}/{len(validation_rows)}")

    probability_array = np.asarray(probabilities, dtype=np.float64)
    confidence = probability_array.max(axis=1)
    correct = np.asarray(predictions) == np.asarray(gold_labels)
    class_report = classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    bootstrap_rng = np.random.default_rng(43000 + train_seed)
    bootstrap_accuracy = []
    bootstrap_macro_f1 = []
    gold_array = np.asarray(gold_labels)
    prediction_array = np.asarray(predictions)
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample_indices = bootstrap_rng.integers(
            0, len(validation_rows), size=len(validation_rows)
        )
        bootstrap_accuracy.append(accuracy_score(
            gold_array[sample_indices], prediction_array[sample_indices]
        ))
        bootstrap_macro_f1.append(f1_score(
            gold_array[sample_indices],
            prediction_array[sample_indices],
            labels=LABELS,
            average="macro",
            zero_division=0,
        ))

    baseline = BASELINE_CE[train_seed]
    metrics = {
        "experiment": "C4_weighted_CE",
        "run_name": run_name,
        "loss_variant": "inverse_frequency_weighted_token_CE",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": train_seed,
        "split_seed": SPLIT_SEED,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "max_length": MAX_LENGTH,
        "class_counts_train": train_counts,
        "class_weights": class_weights,
        "mean_prompt_tokens_before_truncation": float(np.mean(prompt_lengths)),
        "median_prompt_tokens_before_truncation": float(np.median(prompt_lengths)),
        "train_seconds": train_seconds,
        "peak_gpu_memory_reserved_gb": peak_gpu_memory_gb,
        "train_loss": float(train_result.training_loss),
        "accuracy": float(accuracy_score(gold_labels, predictions)),
        "macro_f1_8": float(f1_score(
            gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )),
        "baseline_CE_accuracy_same_seed": baseline["accuracy"],
        "baseline_CE_macro_f1_8_same_seed": baseline["macro_f1_8"],
        "accuracy_delta_vs_CE": float(accuracy_score(gold_labels, predictions) - baseline["accuracy"]),
        "macro_f1_delta_vs_CE": float(f1_score(
            gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
        ) - baseline["macro_f1_8"]),
        "ece_10_bins_raw": expected_calibration_error(confidence, correct),
        "ece_note": "Exploratory, uncalibrated softmax over eight label log-likelihoods.",
        "invalid_count": 0,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": 43000 + train_seed,
        "accuracy_bootstrap_95ci": [
            float(np.quantile(bootstrap_accuracy, 0.025)),
            float(np.quantile(bootstrap_accuracy, 0.975)),
        ],
        "macro_f1_8_bootstrap_95ci": [
            float(np.quantile(bootstrap_macro_f1, 0.025)),
            float(np.quantile(bootstrap_macro_f1, 0.975)),
        ],
        "per_class": class_report,
        "dataset_train_sha256": train_file_hash,
    }
    print(classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        digits=4,
    ))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    pd.DataFrame({
        "sample_id": [f"train:{index}" for index in validation_indices],
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
            "quantization": "4-bit NF4 with double quantization",
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "validation_fraction": VALIDATION_FRACTION,
        }, file, ensure_ascii=False, indent=2)
    return metrics


all_metrics = []
for seed in TRAIN_SEEDS:
    print("\n" + "=" * 72)
    print(f"STARTING C4 WEIGHTED CE, SEED {seed}")
    print("=" * 72)
    run_metrics = run_weighted_seed(seed)
    all_metrics.append(run_metrics)
    summary_rows = [{
        "run_name": item["run_name"],
        "seed": item["seed"],
        "accuracy": item["accuracy"],
        "macro_f1_8": item["macro_f1_8"],
        "accuracy_delta_vs_CE": item["accuracy_delta_vs_CE"],
        "macro_f1_delta_vs_CE": item["macro_f1_delta_vs_CE"],
        "fix_f1": item["per_class"]["fix"]["f1-score"],
        "refactor_f1": item["per_class"]["refactor"]["f1-score"],
        "ece_10_bins_raw": item["ece_10_bins_raw"],
        "train_seconds": item["train_seconds"],
        "peak_gpu_memory_reserved_gb": item["peak_gpu_memory_reserved_gb"],
    } for item in all_metrics]
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_ROOT / "all_experiments_summary.csv", index=False
    )
    del run_metrics
    gc.collect()
    torch.cuda.empty_cache()

print("\nSaved weighted-CE results to:", OUTPUT_ROOT)
print(pd.read_csv(OUTPUT_ROOT / "all_experiments_summary.csv").to_string(index=False))
```

## Đọc kết quả và cổng dừng

- So sánh mỗi weighted run với CE cùng seed: chênh lệch Accuracy/Macro-F1, F1 `fix`/`refactor`, kèm CI/bootstrap và chi phí.
- Chỉ xem weighted CE là ứng viên nếu cải thiện `fix`/`refactor` mà Macro-F1 tổng không suy giảm rõ; với một validation split, kết luận vẫn là pilot.
- Nếu loss NaN, không dự đoán đủ lớp hoặc có output bất thường, dừng nhánh; không đánh giá test.
- `invalid_count = 0` do scoring bị giới hạn tám nhãn. ECE là thô, chưa calibration.
- Sau khi chạy, gửi `all_experiments_summary.csv` cùng ba `metrics.json` và predictions để cập nhật nhật ký. Test chính thức chỉ chạy sau khi khóa lựa chọn.
