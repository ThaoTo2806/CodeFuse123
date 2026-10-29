# Kaggle C2: so sánh nguồn thông tin cho CHCL

## Câu hỏi

Khi phân loại commit, thông tin hữu ích đến từ diff, commit message hay cả hai? So sánh ba nhánh:

- `summary_only`: chỉ dùng `summaries.en`.
- `diff_only`: chỉ dùng `change`.
- `both`: dùng cả diff và commit message.

Mỗi nhánh dùng Qwen2.5-Coder-1.5B-Instruct, QLoRA 4-bit, split stratified seed 42, train seed 0, hai epoch và cùng cách chấm điểm tập đóng 8 nhãn. Không đọc test chính thức. Chỉ thay phần thông tin đầu vào.

**Quan trọng về so sánh công bằng:** C1 compact seed 0 đã dùng chính xác prompt `both` dưới đây, cùng split/config. Có thể dùng nó làm nhánh `both` để tiết kiệm một lượt train, nhưng model revision của C1 chưa được khóa trong metrics. Vì vậy, kết quả ba nhánh chỉ là so sánh pilot nếu không xác nhận được cùng model revision. Nếu cần so sánh chặt, chạy cả ba nhánh C2 với cùng một `MODEL_REVISION` cố định; khi chạy song song, điền cùng commit SHA vào mọi session. Đừng trộn revision khác nhau mà không ghi chú.

## Cách chạy

1. Trên Kaggle, attach dataset HQCM gốc có `train.json` và bật GPU.
2. Cài thư viện nếu cần, rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

3. Chạy cell ba lần, mỗi Kaggle session một `INPUT_VARIANT`: `summary_only`, `diff_only`, `both`. Chạy song song để tiết kiệm thời gian. Nếu dùng kết quả C1 compact cho `both`, chỉ chạy hai variant còn thiếu và ghi rõ revision chưa được khóa.
4. Tải các thư mục output về trước khi Kaggle session hết hạn. Ghi metric vào [nhật ký tiến độ](THEO_DOI_TIEN_DO.md); chỉ chọn cấu hình trên validation.

## Cell

```python
from pathlib import Path
import hashlib
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
INPUT_VARIANT = "summary_only"  # Run separate sessions: summary_only, diff_only, both.
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = None  # Set the same resolved commit SHA in every comparison session.
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
    candidates = sorted({
        path.parent
        for path in Path("/kaggle/input").rglob("train.json")
        if (path.parent / "test.json").is_file()
    })
    if len(candidates) != 1:
        raise RuntimeError(
            "Set DATA_DIR to the intended folder containing the HQCM train.json and test.json."
        )
    DATA_DIR = candidates[0]
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
if any(not row.get("summaries", {}).get("en") for row in raw_rows):
    raise ValueError("Every training row must have a non-empty summaries.en value.")

RUN_NAME = f"c2_{INPUT_VARIANT}_seed{TRAIN_SEED}"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(TRAIN_SEED)
random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)

def build_prompt(row):
    instruction = (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
    )
    if INPUT_VARIANT == "summary_only":
        return instruction + f"Commit message:\n{row['summaries']['en']}\nType:"
    if INPUT_VARIANT == "diff_only":
        return instruction + f"Diff:\n{row['change']}\nType:"
    return (
        instruction
        + f"Diff:\n{row['change']}\n"
        + f"Commit message:\n{row['summaries']['en']}\nType:"
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
print(
    f"Variant: {INPUT_VARIANT} | train={len(train_rows)} | "
    f"validation={len(validation_rows)}"
)

model_kwargs = {"trust_remote_code": False}
if MODEL_REVISION:
    model_kwargs["revision"] = MODEL_REVISION
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **model_kwargs)
resolved_revision = tokenizer.init_kwargs.get("_commit_hash")
if not resolved_revision:
    raise RuntimeError(
        "Could not resolve the model revision. Set MODEL_REVISION to a pinned commit SHA."
    )
print("Resolved model revision:", resolved_revision)
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
torch.cuda.reset_peak_memory_stats()
train_started = time.perf_counter()
train_result = trainer.train()
train_seconds = time.perf_counter() - train_started
peak_gpu_memory_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
adapter_output = OUTPUT_DIR / "adapter"
trainer.save_model(str(adapter_output))
tokenizer.save_pretrained(str(adapter_output))
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
    scores = score_prompt(prompt)
    class_probabilities = torch.softmax(scores, dim=0).numpy()
    predictions.append(LABELS[int(class_probabilities.argmax())])
    probabilities.append(class_probabilities)
    if index % 50 == 0 or index == len(validation_rows):
        print(f"Scored validation {index}/{len(validation_rows)}")

probabilities = np.asarray(probabilities, dtype=np.float64)
confidence = probabilities.max(axis=1)
correct = np.asarray(predictions) == np.asarray(gold_labels)
bin_edges = np.linspace(0.0, 1.0, 11)
ece = 0.0
for bin_index in range(10):
    lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
    in_bin = (
        (confidence >= lower) & (confidence <= upper)
        if bin_index == 9
        else (confidence >= lower) & (confidence < upper)
    )
    if in_bin.any():
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())

metrics = {
    "experiment": "C2_input_ablation",
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
    "peak_gpu_memory_reserved_gb": peak_gpu_memory_gb,
    "train_loss": float(train_result.training_loss),
    "accuracy": float(accuracy_score(gold_labels, predictions)),
    "macro_f1_8": float(f1_score(
        gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
    )),
    "ece_10_bins_raw": float(ece),
    "ece_note": "Exploratory, uncalibrated softmax over eight label log-likelihoods.",
    "invalid_count": 0,
    "inference": "Constrained to the fixed eight-label set using summed label-plus-EOS log-likelihood.",
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

prediction_frame = pd.DataFrame({
    "sample_id": [f"train:{index}" for index in validation_indices],
    "gold_label": gold_labels,
    "predicted_label": predictions,
    "confidence": confidence,
})
prediction_frame.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
with (OUTPUT_DIR / "split_indices.json").open("w", encoding="utf-8") as file:
    json.dump({
        "split_seed": SPLIT_SEED,
        "train_indices": [int(index) for index in train_indices],
        "validation_indices": [int(index) for index in validation_indices],
    }, file, ensure_ascii=False)

manifest = {
    **metrics,
    "dataset_dir": str(DATA_DIR),
    "dataset_train_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
    "dtype": "float16",
    "quantization": "4-bit NF4 with double quantization",
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "validation_fraction": VALIDATION_FRACTION,
}
with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(manifest, file, ensure_ascii=False, indent=2)
print("Saved C2 artifacts to:", OUTPUT_DIR)
```

## Đọc kết quả

- So sánh Accuracy, Macro-F1 (8 nhãn), F1 từng lớp, ECE thô, thời gian train và VRAM. ECE chưa được hiệu chỉnh.
- `invalid_count = 0` theo thiết kế chấm điểm tám nhãn, không chứng minh sinh tự do không thể tạo nhãn sai.
- `summary_only` thấp hơn không có nghĩa message vô dụng nếu message là kết quả cần phân loại; diễn giải đúng là model cần diff, message hay cả hai trong protocol này.
- Đây vẫn là một seed. Sau khi chọn nhánh trên validation mới lập kế hoạch seed 1/2 và đánh giá test một lần cho cấu hình đã chốt.
- Giữ lại `metrics.json`, `validation_predictions.csv`, `split_indices.json`, `run_manifest.json` và thư mục `adapter`.
