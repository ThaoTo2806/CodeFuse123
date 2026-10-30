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
from huggingface_hub import HfApi
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
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

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("/kaggle/working/CodeFuse123/dataset")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_ROOT = Path("/kaggle/working/hqcm_c2_input_ablation")

INPUT_VARIANTS = ["summary_only", "diff_only", "both"]

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

LABELS = [
    "style", "docs", "test", "build",
    "cicd", "fix", "feat", "refactor",
]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("Hãy bật GPU trong Kaggle trước khi chạy.")

TRAIN_PATH = DATA_DIR / "train.json"
if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy {TRAIN_PATH}")

with TRAIN_PATH.open(encoding="utf-8") as f:
    raw_rows = json.load(f)

if len(raw_rows) != 4611:
    raise ValueError(f"Expected 4611 rows, found {len(raw_rows)}")

if any(row.get("type") not in LABELS for row in raw_rows):
    raise ValueError("Dataset chứa nhãn không hợp lệ.")

if any(not row.get("summaries", {}).get("en") for row in raw_rows):
    raise ValueError("Có mẫu thiếu summaries.en.")

# ============================================================
# PIN MODEL REVISION ONCE FOR ALL EXPERIMENTS
# ============================================================
print("Resolving model revision...")

model_info = HfApi().model_info(
    repo_id=MODEL_ID,
    revision="main",
)
MODEL_REVISION = model_info.sha

if not MODEL_REVISION:
    raise RuntimeError("Không lấy được model commit SHA.")

print("Model:", MODEL_ID)
print("Pinned revision:", MODEL_REVISION)

# ============================================================
# FIXED DATA SPLIT FOR ALL THREE EXPERIMENTS
# ============================================================
train_indices, validation_indices = train_test_split(
    np.arange(len(raw_rows)),
    test_size=VALIDATION_FRACTION,
    random_state=SPLIT_SEED,
    stratify=[row["type"] for row in raw_rows],
)

train_rows = [raw_rows[i] for i in train_indices]
validation_rows = [raw_rows[i] for i in validation_indices]

print(
    f"Train={len(train_rows)}, "
    f"validation={len(validation_rows)}"
)

# Save split once so every experiment uses exactly the same samples.
split_path = OUTPUT_ROOT / "split_indices.json"
with split_path.open("w", encoding="utf-8") as f:
    json.dump({
        "split_seed": SPLIT_SEED,
        "train_indices": [int(i) for i in train_indices],
        "validation_indices": [int(i) for i in validation_indices],
    }, f, indent=2)

dataset_hash = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()

# ============================================================
# PROMPT
# ============================================================
def build_prompt(row, variant):
    instruction = (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
    )

    if variant == "summary_only":
        return (
            instruction
            + f"Commit message:\n{row['summaries']['en']}\nType:"
        )

    if variant == "diff_only":
        return instruction + f"Diff:\n{row['change']}\nType:"

    if variant == "both":
        return (
            instruction
            + f"Diff:\n{row['change']}\n"
            + f"Commit message:\n{row['summaries']['en']}\nType:"
        )

    raise ValueError(f"Unknown variant: {variant}")


# ============================================================
# EVALUATION
# ============================================================
def score_prompt(model, tokenizer, prompt, label_token_ids):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    sequences = [
        prompt_ids + label_token_ids[label]
        for label in LABELS
    ]

    padded_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), padded_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)

    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attention_mask[i, :len(seq)] = 1

    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()

        log_probs = torch.log_softmax(logits, dim=-1)

    scores = []
    prompt_length = len(prompt_ids)

    for i, label in enumerate(LABELS):
        target_ids = label_token_ids[label]

        positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(target_ids) - 1,
            device=device,
        )
        targets = torch.tensor(target_ids, device=device)

        scores.append(
            log_probs[i, positions, targets].sum()
        )

    return torch.stack(scores).cpu()


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================
def run_experiment(variant):
    run_name = f"c2_{variant}_seed{TRAIN_SEED}"
    output_dir = OUTPUT_ROOT / run_name
    metrics_path = output_dir / "metrics.json"

    # Skip only when previous run has a saved result.
    if metrics_path.is_file():
        try:
            previous = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            if (
                previous.get("model_revision") == MODEL_REVISION
                and previous.get("dataset_train_sha256") == dataset_hash
                and previous.get("input_variant") == variant
            ):
                print(f"\nAlready completed: {variant}; skipping.")
                return previous
        except (ValueError, OSError):
            pass

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"STARTING EXPERIMENT: {variant}")
    print("=" * 70)

    set_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

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
        prompt_ids = (
            prompt_ids[-prompt_budget:]
            if prompt_budget > 0 else []
        )

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
        desc=f"Tokenizing {variant}",
    )

    train_dataset = train_dataset.filter(
        lambda row: any(x != -100 for x in row["labels"]),
        desc="Checking target labels",
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
        seed=TRAIN_SEED,
        # Do not pass data_seed here: older Accelerate versions
        # may reject it. The data split above remains fixed.
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
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_reserved() / (1024 ** 3)
    )

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    model.config.use_cache = True
    model.eval()

    label_token_ids = {
        label: tokenizer(
            " " + label,
            add_special_tokens=False,
        )["input_ids"]
        + (
            [tokenizer.eos_token_id]
            if tokenizer.eos_token_id is not None else []
        )
        for label in LABELS
    }

    gold_labels = [row["type"] for row in validation_rows]
    predictions = []
    probabilities = []
    prompt_lengths = []

    for index, row in enumerate(validation_rows, start=1):
        prompt = build_prompt(row, variant)

        prompt_lengths.append(len(tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]))

        scores = score_prompt(
            model, tokenizer, prompt, label_token_ids
        )
        class_probabilities = torch.softmax(
            scores, dim=0
        ).numpy()

        predictions.append(
            LABELS[int(class_probabilities.argmax())]
        )
        probabilities.append(class_probabilities)

        if index % 50 == 0 or index == len(validation_rows):
            print(
                f"[{variant}] Scored "
                f"{index}/{len(validation_rows)}"
            )

    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predictions) == np.asarray(gold_labels)

    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0

    for bin_index in range(10):
        lower, upper = bin_edges[bin_index:bin_index + 2]

        if bin_index == 9:
            in_bin = (
                (confidence >= lower) & (confidence <= upper)
            )
        else:
            in_bin = (
                (confidence >= lower) & (confidence < upper)
            )

        if in_bin.any():
            ece += in_bin.mean() * abs(
                correct[in_bin].mean()
                - confidence[in_bin].mean()
            )

    print(classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        digits=4,
    ))

    metrics = {
        "experiment": "C2_input_ablation",
        "input_variant": variant,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": TRAIN_SEED,
        "split_seed": SPLIT_SEED,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "max_length": MAX_LENGTH,
        "mean_prompt_tokens_before_truncation": float(
            np.mean(prompt_lengths)
        ),
        "median_prompt_tokens_before_truncation": float(
            np.median(prompt_lengths)
        ),
        "train_seconds": train_seconds,
        "peak_gpu_memory_reserved_gb": peak_gpu_memory_gb,
        "train_loss": float(train_result.training_loss),
        "accuracy": float(
            accuracy_score(gold_labels, predictions)
        ),
        "macro_f1_8": float(f1_score(
            gold_labels,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )),
        "ece_10_bins_raw": float(ece),
        "invalid_count": 0,
        "dataset_train_sha256": dataset_hash,
    }

    prediction_frame = pd.DataFrame({
        "sample_id": [
            f"train:{i}" for i in validation_indices
        ],
        "gold_label": gold_labels,
        "predicted_label": predictions,
        "confidence": confidence,
    })
    prediction_frame.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    manifest = {
        **metrics,
        "dataset_dir": str(DATA_DIR),
        "dtype": "float16",
        "quantization": "4-bit NF4 with double quantization",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    }

    with (output_dir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Completed: {variant}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Release GPU memory before loading the next model.
    del trainer, model, base_model, tokenizer
    del train_dataset, data_collator
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


# ============================================================
# RUN ALL THREE SEQUENTIALLY
# ============================================================
all_results = []

for variant in INPUT_VARIANTS:
    try:
        result = run_experiment(variant)
        all_results.append(result)
    except Exception as exc:
        print(f"\nFAILED: {variant}: {type(exc).__name__}: {exc}")
        print("Các thí nghiệm tiếp theo sẽ tiếp tục nếu môi trường còn hoạt động.")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

if all_results:
    summary = pd.DataFrame(all_results)
    summary.to_csv(
        OUTPUT_ROOT / "all_experiments_summary.csv",
        index=False,
    )
    print("\nFINAL SUMMARY")
    print(summary.to_string(index=False))

print("\nOutput directory:", OUTPUT_ROOT)
```

## Đọc kết quả

- So sánh Accuracy, Macro-F1 (8 nhãn), F1 từng lớp, ECE thô, thời gian train và VRAM. ECE chưa được hiệu chỉnh.
- `invalid_count = 0` theo thiết kế chấm điểm tám nhãn, không chứng minh sinh tự do không thể tạo nhãn sai.
- `summary_only` thấp hơn không có nghĩa message vô dụng nếu message là kết quả cần phân loại; diễn giải đúng là model cần diff, message hay cả hai trong protocol này.
- Đây vẫn là một seed. Sau khi chọn nhánh trên validation mới lập kế hoạch seed 1/2 và đánh giá test một lần cho cấu hình đã chốt.
- Giữ lại `metrics.json`, `validation_predictions.csv`, `split_indices.json`, `run_manifest.json` và thư mục `adapter`.
