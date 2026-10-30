# Kaggle C5: xác nhận kết quả qua seed

## Mục tiêu

Kiểm tra xem kết quả C2 có ổn định qua seed hay không. Chạy lại hai nhánh `diff_only` và `both` ở train seed 1 và 2. C2 seed 0 đã có sẵn; sau khi hoàn tất sẽ có ba seed (0/1/2) cho mỗi nhánh.

Mỗi lần chạy chỉ đổi `INPUT_VARIANT` và `TRAIN_SEED`. Giữ nguyên dataset, validation split seed 42, model commit, prompt, max length, LoRA và toàn bộ training config. Cell **không đọc test chính thức**.

## Chuẩn bị và chạy

1. Kaggle: attach đúng dataset HQCM gốc có `train.json`, bật GPU.
2. Cài package nếu cần, rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

3. Chạy cell bốn lần (có thể dùng bốn session):

| Run | `INPUT_VARIANT` | `TRAIN_SEED` |
| :-- | :-------------- | -----------: |
| 1   | `diff_only`     |            1 |
| 2   | `diff_only`     |            2 |
| 3   | `both`          |            1 |
| 4   | `both`          |            2 |

Train seed 0 đã chạy trong C2; không cần chạy lại. Giữ `MODEL_REVISION` chính xác như trong cell. Mỗi run tạo thư mục riêng trong `/kaggle/working/hqcm_c5_seed_confirmation/`. Tải kết quả về trước khi session hết hạn.

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
# CONFIGURATION
# ============================================================

DATA_DIR = Path("/kaggle/working/CodeFuse123/dataset")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Pin the exact same Hugging Face model revision used in C2/C5.
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"

OUTPUT_ROOT = Path("/kaggle/working/hqcm_c5_seed_confirmation")

# Required execution order:
# diff_only_seed1 -> diff_only_seed2 -> both_seed1 -> both_seed2
EXPERIMENTS = [
    ("diff_only", 1),
    ("diff_only", 2),
    ("both", 1),
    ("both", 2),
]

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

EXPECTED_TRAIN_SHA256 = (
    "a6430d65f194f5656b4704848d56c7e924bd99532ec78361b18c387a2f44c543"
)

LABELS = [
    "style",
    "docs",
    "test",
    "build",
    "cicd",
    "fix",
    "feat",
    "refactor",
]


# ============================================================
# ENVIRONMENT AND DATA VALIDATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before running this script.")

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

TRAIN_PATH = DATA_DIR / "train.json"
if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Raw train.json not found: {TRAIN_PATH}")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

train_file_hash = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()

if train_file_hash != EXPECTED_TRAIN_SHA256:
    raise ValueError(
        "The train.json SHA-256 does not match the dataset used in C2. "
        "Use the exact same raw dataset. "
        f"Expected: {EXPECTED_TRAIN_SHA256}; found: {train_file_hash}"
    )

with TRAIN_PATH.open("r", encoding="utf-8") as file:
    raw_rows = json.load(file)

if len(raw_rows) != 4611:
    raise ValueError(
        f"Expected 4,611 training rows; found {len(raw_rows)}."
    )

if any(row.get("type") not in LABELS for row in raw_rows):
    raise ValueError(
        "The dataset contains labels outside the fixed eight-label set."
    )

if any(not row.get("change") for row in raw_rows):
    raise ValueError("At least one sample has an empty/missing code diff.")

if any(not row.get("summaries", {}).get("en") for row in raw_rows):
    raise ValueError("At least one sample has a missing summaries.en field.")

print("=" * 75)
print("C5: MULTI-SEED INPUT ABLATION")
print("=" * 75)
print("Dataset:", DATA_DIR)
print("Dataset SHA-256:", train_file_hash)
print("Model:", MODEL_ID)
print("Pinned revision:", MODEL_REVISION)
print("Execution order:", " -> ".join(
    f"{variant}_seed{seed}" for variant, seed in EXPERIMENTS
))
print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# FIXED TRAIN / VALIDATION SPLIT
# Identical samples for all four runs.
# ============================================================

train_indices, validation_indices = train_test_split(
    np.arange(len(raw_rows)),
    test_size=VALIDATION_FRACTION,
    random_state=SPLIT_SEED,
    stratify=[row["type"] for row in raw_rows],
)

train_rows = [raw_rows[int(index)] for index in train_indices]
validation_rows = [raw_rows[int(index)] for index in validation_indices]

split_path = OUTPUT_ROOT / "split_indices.json"
split_payload = {
    "split_seed": SPLIT_SEED,
    "validation_fraction": VALIDATION_FRACTION,
    "train_indices": [int(index) for index in train_indices],
    "validation_indices": [int(index) for index in validation_indices],
}
with split_path.open("w", encoding="utf-8") as file:
    json.dump(split_payload, file, ensure_ascii=False, indent=2)

print(f"Train samples: {len(train_rows)}")
print(f"Validation samples: {len(validation_rows)}")
print("Saved fixed split:", split_path)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(row, input_variant):
    instruction = (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
    )

    if input_variant == "diff_only":
        return instruction + f"Diff:\n{row['change']}\nType:"

    if input_variant == "both":
        return (
            instruction
            + f"Diff:\n{row['change']}\n"
            + f"Commit message:\n{row['summaries']['en']}\n"
            + "Type:"
        )

    raise ValueError(f"Unsupported input variant: {input_variant}")


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_experiment(input_variant, train_seed):
    run_name = f"c5_{input_variant}_seed{train_seed}"
    output_dir = OUTPUT_ROOT / run_name
    metrics_path = output_dir / "metrics.json"

    # Skip a run only if its saved result matches this exact setup.
    if metrics_path.is_file():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            is_matching_run = (
                previous.get("experiment") == "C5_seed_confirmation"
                and previous.get("input_variant") == input_variant
                and previous.get("seed") == train_seed
                and previous.get("model_revision") == MODEL_REVISION
                and previous.get("dataset_train_sha256") == train_file_hash
                and previous.get("split_seed") == SPLIT_SEED
            )
            if is_matching_run:
                print(f"\nAlready completed: {run_name}; skipping.")
                return previous
        except (OSError, ValueError, TypeError):
            pass

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"STARTING: {run_name}")
    print("=" * 75)

    # Reset all random generators for reproducibility.
    set_seed(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

    # --------------------------------------------------------
    # Load tokenizer from the pinned Hugging Face revision.
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )

    # Hugging Face receives the exact immutable commit SHA through
    # revision=MODEL_REVISION above. Some Transformers/tokenizer versions
    # do not expose the resolved SHA in tokenizer.init_kwargs, so do not
    # treat a missing _commit_hash there as a loading failure.
    #
    # The same pinned revision is also passed to AutoModelForCausalLM below.
    # Record the requested immutable revision in the experiment metadata.
    resolved_revision = MODEL_REVISION
    print("Using pinned Hugging Face revision:", resolved_revision)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Tokenization and causal-LM labels.
    # Loss is calculated only on the target label tokens.
    # --------------------------------------------------------
    def encode_example(row):
        prompt_ids = tokenizer(
            build_prompt(row, input_variant),
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

        # Keep the end of the prompt so that the "Type:" instruction
        # and the most recent part of the input are retained.
        prompt_ids = (
            prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
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
        desc=f"Tokenizing {input_variant} seed {train_seed}",
    )
    train_dataset = train_dataset.filter(
        lambda row: any(label != -100 for label in row["labels"]),
        desc="Checking target labels",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Load quantized base model and attach LoRA adapters.
    # --------------------------------------------------------
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
        trust_remote_code=False,
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
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Training.
    # --------------------------------------------------------
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

    # Save adapter and tokenizer.
    adapter_output = output_dir / "adapter"
    trainer.save_model(str(adapter_output))
    tokenizer.save_pretrained(str(adapter_output))

    model.config.use_cache = True
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    # --------------------------------------------------------
    # Candidate label token sequences.
    # --------------------------------------------------------
    label_token_ids = {
        label: tokenizer(
            " " + label,
            add_special_tokens=False,
        )["input_ids"]
        + (
            [tokenizer.eos_token_id]
            if tokenizer.eos_token_id is not None
            else []
        )
        for label in LABELS
    }

    # --------------------------------------------------------
    # Score all eight candidate labels for one prompt.
    # --------------------------------------------------------
    def score_prompt(prompt):
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

        padded_length = max(len(sequence) for sequence in sequences)

        input_ids = torch.full(
            (len(sequences), padded_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row_index, sequence in enumerate(sequences):
            input_ids[row_index, :len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
            )
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
            targets = torch.tensor(
                target_ids,
                dtype=torch.long,
                device=input_device,
            )

            scores.append(
                log_probabilities[row_index, positions, targets].sum()
            )

        return torch.stack(scores).cpu()

    # --------------------------------------------------------
    # Validation prediction.
    # --------------------------------------------------------
    gold_labels = [row["type"] for row in validation_rows]
    predictions = []
    probabilities = []
    prompt_lengths = []

    for index, row in enumerate(validation_rows, start=1):
        prompt = build_prompt(row, input_variant)

        prompt_lengths.append(
            len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
        )

        scores = score_prompt(prompt)
        class_probabilities = torch.softmax(scores, dim=0).numpy()

        predictions.append(
            LABELS[int(class_probabilities.argmax())]
        )
        probabilities.append(class_probabilities)

        if index % 50 == 0 or index == len(validation_rows):
            print(
                f"[{run_name}] Scored "
                f"{index}/{len(validation_rows)}"
            )

    probability_array = np.asarray(probabilities, dtype=np.float64)
    confidence = probability_array.max(axis=1)
    correct = np.asarray(predictions) == np.asarray(gold_labels)

    # --------------------------------------------------------
    # Expected Calibration Error (ECE), 10 equal-width bins.
    # --------------------------------------------------------
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0

    for bin_index in range(10):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

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

    # --------------------------------------------------------
    # Classification report and bootstrap 95% confidence
    # intervals for Accuracy and Macro-F1.
    # --------------------------------------------------------
    class_report = classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    bootstrap_rng = np.random.default_rng(42000 + train_seed)
    bootstrap_accuracy = []
    bootstrap_macro_f1 = []

    gold_array = np.asarray(gold_labels)
    prediction_array = np.asarray(predictions)

    for _ in range(BOOTSTRAP_RESAMPLES):
        sample_indices = bootstrap_rng.integers(
            0,
            len(validation_rows),
            size=len(validation_rows),
        )

        bootstrap_accuracy.append(
            accuracy_score(
                gold_array[sample_indices],
                prediction_array[sample_indices],
            )
        )

        bootstrap_macro_f1.append(
            f1_score(
                gold_array[sample_indices],
                prediction_array[sample_indices],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )

    accuracy = float(accuracy_score(gold_labels, predictions))
    macro_f1 = float(
        f1_score(
            gold_labels,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    metrics = {
        "experiment": "C5_seed_confirmation",
        "run_name": run_name,
        "input_variant": input_variant,
        "model_id": MODEL_ID,
        "model_revision": resolved_revision,
        "seed": train_seed,
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
        "train_seconds": float(train_seconds),
        "peak_gpu_memory_reserved_gb": float(peak_gpu_memory_gb),
        "train_loss": float(train_result.training_loss),
        "accuracy": accuracy,
        "macro_f1_8": macro_f1,
        "ece_10_bins_raw": float(ece),
        "ece_note": (
            "Exploratory, uncalibrated softmax over eight label "
            "log-likelihoods."
        ),
        "invalid_count": 0,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": 42000 + train_seed,
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

    print("\nClassification report:")
    print(
        classification_report(
            gold_labels,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
            digits=4,
        )
    )

    print("\nMetrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # --------------------------------------------------------
    # Save predictions, metrics, split, and run manifest.
    # --------------------------------------------------------
    prediction_frame = pd.DataFrame(
        {
            "sample_id": [
                f"train:{int(index)}" for index in validation_indices
            ],
            "gold_label": gold_labels,
            "predicted_label": predictions,
            "confidence": confidence,
        }
    )
    prediction_frame.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with (output_dir / "split_indices.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(split_payload, file, ensure_ascii=False, indent=2)

    manifest = {
        **metrics,
        "dataset_dir": str(DATA_DIR),
        "dtype": "float16",
        "quantization": "4-bit NF4 with double quantization",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "validation_fraction": VALIDATION_FRACTION,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }

    with (output_dir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"\nCompleted: {run_name}")
    print("Saved artifacts to:", output_dir)

    # --------------------------------------------------------
    # Release GPU memory before the next experiment.
    # --------------------------------------------------------
    del trainer
    del model
    del base_model
    del tokenizer
    del train_dataset
    del data_collator
    del train_result

    gc.collect()
    torch.cuda.empty_cache()

    return metrics


# ============================================================
# RUN ALL FOUR EXPERIMENTS SEQUENTIALLY
# ============================================================

all_results = []

for input_variant, train_seed in EXPERIMENTS:
    try:
        result = run_experiment(input_variant, train_seed)
        all_results.append(result)

    except Exception as exc:
        print(
            f"\nFAILED: {input_variant}_seed{train_seed}: "
            f"{type(exc).__name__}: {exc}"
        )
        print("Continuing to the next experiment if the runtime is usable.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        continue


# ============================================================
# FINAL SUMMARY
# Includes previously completed runs when their metrics files
# are available, so rerunning the script can still produce a
# complete four-run summary.
# ============================================================

summary_rows = []

for input_variant, train_seed in EXPERIMENTS:
    metrics_path = (
        OUTPUT_ROOT
        / f"c5_{input_variant}_seed{train_seed}"
        / "metrics.json"
    )

    if not metrics_path.is_file():
        continue

    try:
        saved_metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        if (
            saved_metrics.get("model_revision") == MODEL_REVISION
            and saved_metrics.get("dataset_train_sha256") == train_file_hash
            and saved_metrics.get("input_variant") == input_variant
            and saved_metrics.get("seed") == train_seed
            and saved_metrics.get("split_seed") == SPLIT_SEED
        ):
            summary_rows.append(saved_metrics)

    except (OSError, ValueError, TypeError):
        print("Could not read metrics:", metrics_path)


if summary_rows:
    summary = pd.DataFrame(summary_rows)

    preferred_columns = [
        "run_name",
        "input_variant",
        "seed",
        "accuracy",
        "macro_f1_8",
        "ece_10_bins_raw",
        "train_loss",
        "train_seconds",
        "peak_gpu_memory_reserved_gb",
        "accuracy_bootstrap_95ci",
        "macro_f1_8_bootstrap_95ci",
    ]
    available_columns = [
        column for column in preferred_columns if column in summary.columns
    ]

    summary_path = OUTPUT_ROOT / "all_experiments_summary.csv"
    summary[available_columns].to_csv(summary_path, index=False)

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)
    print(summary[available_columns].to_string(index=False))
    print("\nSaved summary:", summary_path)

else:
    print("\nNo completed experiment metrics were found.")

print("\nOutput directory:", OUTPUT_ROOT)
```

## Khi có đủ bốn kết quả

Gửi lại bốn `metrics.json` (hai nhánh × seed 1/2) và `validation_predictions.csv`. So sánh từng nhánh qua ba seed (seed 0 từ C2 và seed 1/2 từ C5): trung bình/độ lệch chuẩn Accuracy và Macro-F1, khoảng bootstrap trên validation, F1 từng lớp và chênh lệch paired `both - diff_only` trên cùng mẫu. Không diễn giải ECE là calibration đã xác nhận; không đánh giá/chọn bằng test.
