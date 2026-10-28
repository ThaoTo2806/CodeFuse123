# Kaggle - C2: Ablation nguồn thông tin đầu vào

## Mục tiêu

Xác định model dùng được thông tin nào để phân loại commit: summary, diff hay cả hai. Chạy ba Kaggle session song song, mỗi session dùng cùng model, split seed, training seed và cấu hình; chỉ đổi `INPUT_VARIANT`:

- `summary_only`
- `diff_only`
- `both`

Cell dùng dataset train chính thức, tách validation stratified 10% từ train và không đọc test chính thức 518 mẫu. `both` dùng prompt compact đã cho kết quả pilot tốt trong C1. Kết quả C1 hiện chỉ có một seed, nên giữ seed `0` để ablation công bằng; C2 cũng là thăm dò cho bước chọn cấu hình, không phải kết luận cuối.

## Chuẩn bị

1. Attach dataset gốc HQCM có `train.json` và `test.json`.
2. Bật GPU. Model Qwen2.5-Coder-1.5B-Instruct là public; không cần Hugging Face token nếu Kaggle cho tải model.
3. Nếu thiếu package, cài riêng rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

## Cell Kaggle

Mỗi nhánh lưu vào folder riêng. Tạo ba session từ cell này, đổi `INPUT_VARIANT` tương ứng, còn mọi giá trị khác giữ nguyên.

```python
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
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

DATA_DIR = None  # Example: Path("/kaggle/input/hqcm/dataset")
INPUT_VARIANT = "both"  # Run separate sessions: summary_only, diff_only, both.
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = None  # Pin the resolved commit SHA when repeating the experiment.
OUTPUT_ROOT = Path("/kaggle/working/hqcm_c2_input_ablation")
MAX_LENGTH = 512
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 42
TRAIN_SEED = 0
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]

if INPUT_VARIANT not in {"summary_only", "diff_only", "both"}:
    raise ValueError("INPUT_VARIANT must be summary_only, diff_only, or both.")
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before loading the quantized model.")
if DATA_DIR is None:
    data_dirs = sorted({
        path.parent
        for path in Path("/kaggle/input").rglob("train.json")
        if (path.parent / "test.json").is_file()
    })
    if len(data_dirs) != 1:
        raise RuntimeError("Set DATA_DIR to the intended folder containing raw train.json and test.json.")
    DATA_DIR = data_dirs[0]
else:
    DATA_DIR = Path(DATA_DIR)

TRAIN_PATH = DATA_DIR / "train.json"
if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Raw train.json not found: {TRAIN_PATH}")
with TRAIN_PATH.open(encoding="utf-8") as file:
    raw_rows = json.load(file)
if len(raw_rows) != 4611:
    raise ValueError(f"Expected 4,611 official training rows; found {len(raw_rows)}.")
if any(row.get("type") not in LABELS for row in raw_rows):
    raise ValueError("The training data contains labels outside the fixed eight-label set.")

RUN_NAME = f"c2_{INPUT_VARIANT}_seed{TRAIN_SEED}"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(TRAIN_SEED)
random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)


def build_prompt(row):
    summary = str(row["summaries"]["en"]).strip()
    diff = str(row["change"])
    instruction = (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
    )
    if INPUT_VARIANT == "summary_only":
        return instruction + f"Commit message:\n{summary}\nType:"
    if INPUT_VARIANT == "diff_only":
        return instruction + f"Diff:\n{diff}\nType:"
    return (
        instruction
        + f"Diff:\n{diff}\n"
        + f"Commit message:\n{summary}\n"
        + "Type:"
    )


train_indices, validation_indices = train_test_split(
    np.arange(len(raw_rows)),
    test_size=VALIDATION_FRACTION,
    random_state=SPLIT_SEED,
    stratify=[row["type"] for row in raw_rows],
)
train_rows = [raw_rows[index] for index in train_indices]
validation_rows = [raw_rows[index] for index in validation_indices]
print(f"Dataset: {DATA_DIR}")
print(f"Variant: {INPUT_VARIANT} | train={len(train_rows)} | validation={len(validation_rows)}")

tokenizer_kwargs = {"trust_remote_code": False}
if MODEL_REVISION:
    tokenizer_kwargs["revision"] = MODEL_REVISION
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
resolved_revision = tokenizer.init_kwargs.get("_commit_hash")
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
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


train_dataset = Dataset.from_list(train_rows)
train_dataset = train_dataset.map(
    encode_example,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training split",
)
train_dataset = train_dataset.filter(
    lambda row: any(label != -100 for label in row["labels"]),
    desc="Dropping rows without target tokens",
)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=resolved_revision,
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
    output_dir=str(OUTPUT_DIR / "trainer"),
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
    seed=TRAIN_SEED,
    data_seed=SPLIT_SEED,
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
train_started = time.perf_counter()
train_result = trainer.train()
train_seconds = time.perf_counter() - train_started
ADAPTER_OUTPUT = OUTPUT_DIR / "adapter"
trainer.save_model(str(ADAPTER_OUTPUT))
tokenizer.save_pretrained(str(ADAPTER_OUTPUT))
model.config.use_cache = True
model.eval()
input_device = model.get_input_embeddings().weight.device

label_token_ids = {
    label: tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
    for label in LABELS
}


def score_prompt(prompt):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]
    sequences = [prompt_ids + label_token_ids[label] for label in LABELS]
    padded_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), padded_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
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


def expected_calibration_error(confidence_values, correct_values, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index in range(n_bins):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        in_bin = (
            (confidence_values >= lower) & (confidence_values <= upper)
            if bin_index == n_bins - 1
            else (confidence_values >= lower) & (confidence_values < upper)
        )
        if in_bin.any():
            ece += in_bin.mean() * abs(
                correct_values[in_bin].mean() - confidence_values[in_bin].mean()
            )
    return float(ece)


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
    class_probabilities = torch.softmax(score_prompt(prompt), dim=0).numpy()
    predictions.append(LABELS[int(class_probabilities.argmax())])
    probabilities.append(class_probabilities)
    if index % 50 == 0 or index == len(validation_rows):
        print(f"Scored validation {index}/{len(validation_rows)}")

probabilities = np.asarray(probabilities, dtype=np.float64)
confidence = probabilities.max(axis=1)
correct = np.asarray(predictions) == np.asarray(gold_labels)
metrics = {
    "input_variant": INPUT_VARIANT,
    "model_id": MODEL_ID,
    "model_revision": resolved_revision,
    "seed": TRAIN_SEED,
    "split_seed": SPLIT_SEED,
    "train_samples": len(train_rows),
    "validation_samples": len(validation_rows),
    "max_length": MAX_LENGTH,
    "mean_prompt_tokens_before_truncation": float(np.mean(prompt_lengths)),
    "median_prompt_tokens_before_truncation": float(np.median(prompt_lengths)),
    "train_seconds": train_seconds,
    "train_loss": float(train_result.training_loss),
    "accuracy": float(accuracy_score(gold_labels, predictions)),
    "macro_f1_8": float(f1_score(
        gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
    )),
    "ece_10_bins_raw": expected_calibration_error(confidence, correct),
    "ece_note": "Exploratory, uncalibrated softmax over eight label log-likelihoods.",
    "invalid_count": 0,
    "inference": "Constrained to eight labels using summed label-plus-EOS log-likelihood.",
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
    "sample_id": [f"train:{int(index)}" for index in validation_indices],
    "gold_label": gold_labels,
    "predicted_label": predictions,
    "confidence": confidence,
}).to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
with (OUTPUT_DIR / "split_indices.json").open("w", encoding="utf-8") as file:
    json.dump({
        "split_seed": SPLIT_SEED,
        "train_indices": [int(index) for index in train_indices],
        "validation_indices": [int(index) for index in validation_indices],
    }, file, ensure_ascii=False)
print("Saved C2 artifacts to:", OUTPUT_DIR)
```

## Đọc và so sánh

- So sánh Accuracy, Macro-F1, F1 từng lớp, ECE thô, độ dài prompt và thời gian train trên validation.
- `summary_only`, `diff_only`, `both` dùng cùng split và seed; `both` phải tương ứng với prompt compact đã chạy ở C1. Nếu kết quả `both` tái hiện khác C1, kiểm tra revision/tokenizer, thư viện, adapter config và prompt string trước khi giải thích hiệu ứng.
- ECE trong cell là exploratory, chưa temperature-scale. Không calibration trên validation này rồi báo điểm đó như đánh giá độc lập; cần thêm calibration protocol riêng.
- Không dùng các kết quả một seed để khẳng định cải thiện. Dùng chúng để chọn input condition, rồi chạy tối thiểu ba seed cho các cấu hình chính.
