# Kaggle - C1: Prompt CHCL gốc so với prompt rút gọn

## Mục tiêu

So sánh ảnh hưởng của template CHCL dài với prompt ngắn trên cùng model nhỏ, split, seed và cấu hình huấn luyện. Cell chỉ dùng `train.json` gốc, tự tách validation có stratify; không đọc hay đánh giá test chính thức 518 mẫu. Chạy hai Kaggle session song song: một session với `PROMPT_VARIANT = "original"`, session còn lại với `PROMPT_VARIANT = "compact"`.

Đây là ablation về prompt, không phải bằng chứng rằng input ngắn hơn luôn tốt hơn. Cell lưu token trung bình, thời gian train, metric validation, dự đoán và manifest để đối chiếu.

## Chuẩn bị

1. Attach dataset gốc HQCM có `train.json`.
2. Bật GPU. Model Qwen2.5-Coder-1.5B-Instruct là public; không cần Hugging Face token nếu Kaggle cho tải model.
3. Nếu thiếu package, cài riêng rồi restart kernel:

```python
%pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

## Cách chạy hai nhánh

Trong một Kaggle notebook, đặt `PROMPT_VARIANT = "original"`. Tạo notebook/session thứ hai từ cùng cell và chỉ đổi thành `"compact"`. Giữ các hằng số khác nguyên vẹn. Mỗi session tạo output folder riêng; cuối cùng tải hai thư mục kết quả và so sánh `metrics.json`, `run_manifest.json`.

## Cell

````python
from pathlib import Path
import json
import random
import re
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
PROMPT_VARIANT = "original"  # Run one session per value: "original" and "compact".
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = None  # Pin the resolved commit SHA when repeating the experiment.
OUTPUT_ROOT = Path("/kaggle/working/hqcm_c1_prompt_ablation")
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
EVAL_BATCH_SIZE = 8
BOOTSTRAP_RESAMPLES = 2000
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]

if PROMPT_VARIANT not in {"original", "compact"}:
    raise ValueError("PROMPT_VARIANT must be 'original' or 'compact'.")
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before loading the quantized model.")
if DATA_DIR is None:
    candidates = sorted({
        path.parent
        for path in Path("/kaggle/input").rglob("train.json")
        if (path.parent / "test.json").is_file()
    })
    if len(candidates) != 1:
        raise RuntimeError("Set DATA_DIR to the intended folder containing train.json and test.json.")
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

RUN_NAME = f"c1_{PROMPT_VARIANT}_seed{TRAIN_SEED}"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(TRAIN_SEED)
random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)

ORIGINAL_TEMPLATE = """\
A git commit can typically be classified into specific categories by examining its code changes and commit message. These categories include:

- "style": Changes that solely improve the code's formatting and appearance without affecting functionality (e.g., adjusting whitespace, fixing indentation, cleaning up code formatting).
- "docs": Updates or improvements to documentation, which may include inline code comments, README files, or any other type of documentation associated with the project.
- "test": Modifications exclusively related to test code, like the addition of new tests or the correction and improvement of existing tests.
- "build": Changes that affect the build system or tools (like Gulp, Broccoli, NPM) or alterations to external dependencies (e.g., library or package updates).
- "cicd": Tweaks to configuration files or scripts used in Continuous Integration/Continuous Deployment (CI/CD) systems, such as Travis CI or CircleCI configurations.
- "fix": Code amendments that focus on rectifying errors, fixing bugs, or patching security vulnerabilities.
- "feat": Commits that introduce new features or capabilities to the project, such as new classes, functions, or methods.
- "refactor": Changes that reorganize and clean up the codebase without modifying its external behavior or outputs, improving readability and maintainability.

For a given git commit, we can inspect its code difference (diff) and the associated commit message to determine its type. Below is the diff for a specific git commit:

```
__DIFF_PLACEHOLDER__
```

Accompanying this code diff is its commit message:

```
__MESSAGE_PLACEHOLDER__
```

Given this information, the git commit can be categorized as type: """


def build_prompt(row):
    if PROMPT_VARIANT == "original":
        return ORIGINAL_TEMPLATE.replace(
            "__DIFF_PLACEHOLDER__", row["change"]
        ).replace("__MESSAGE_PLACEHOLDER__", row["summaries"]["en"])
    return (
        "Classify this commit into exactly one label: "
        "style, docs, test, build, cicd, fix, feat, refactor.\n"
        f"Diff:\n{row['change']}\n"
        f"Commit message:\n{row['summaries']['en']}\n"
        "Type:"
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
print(f"Variant: {PROMPT_VARIANT} | train={len(train_rows)} | validation={len(validation_rows)}")

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
    "prompt_variant": PROMPT_VARIANT,
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
pd.DataFrame({
    "sample_id": [f"train:{index}" for index in validation_indices],
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
print("Saved C1 artifacts to:", OUTPUT_DIR)
````

## Kết quả C3 đã chạy: 7B constrained scoring

Đã chấm 518 mẫu test chính thức bằng adapter Run 1 và base model `meta-llama/Llama-2-7b-hf`, revision `01c7f73d771dfac7d292323805ebc428287df4f9`. Kết quả do người dùng chạy và cung cấp ngày 28/10/2026:

| Metric                   | C3 constrained scoring |   Run 1 reproduction |                            Chênh lệch |
| :----------------------- | ---------------------: | -------------------: | ------------------------------------: |
| Accuracy                 |                 0,7799 |               0,7761 |                               +0,0039 |
| Macro-F1 (8 nhãn)        |                 0,8384 |               0,8410 |                               -0,0026 |
| Macro-Precision (8 nhãn) |                 0,8402 |               0,8394 |                               +0,0008 |
| Macro-Recall (8 nhãn)    |                 0,8393 |               0,8458 |                               -0,0065 |
| Invalid                  |                  0/518 |                1/518 | Giảm do decoder chỉ chọn trong 8 nhãn |
| ECE thô, 10 bins         |                 0,1948 | Chưa có số tương ứng |                       Chưa hiệu chỉnh |

F1 theo lớp lần lượt là: `style` 0,8182; `docs` 0,8974; `test` 0,8687; `build` 0,9167; `cicd` 0,9524; `fix` 0,7179; `feat` 0,8000; `refactor` 0,7358. Khoảng bootstrap 95% là 0,7432–0,8166 cho Accuracy và 0,7947–0,8752 cho Macro-F1.

**Nhận xét.** Accuracy/Macro-F1 gần Run 1, nhưng đây chưa phải so sánh hoàn toàn cùng protocol: Run 1 dùng generation tự do còn C3 chấm log-likelihood của tám nhãn kèm EOS. Vì vậy không kết luận C3 cải thiện chất lượng chỉ từ chênh lệch này. Invalid bằng 0 là đảm bảo của decoder tập đóng. ECE 0,1948 còn đáng lưu ý nhưng chỉ là ECE thô từ softmax các điểm log-likelihood, chưa temperature-scale; cần tập validation để calibration, tuyệt đối không fit trên test. Kết quả theo lớp vẫn cho thấy `fix` 0,7179 và `refactor` 0,7358 là hai lớp cần ưu tiên ở C4.

Artifact được cell C3 lưu dưới `/kaggle/working/hqcm_huong_b_c3`: `c3_metrics.json` và `c3_test_scores.csv`. Cần đưa artifact/manifest vào Kaggle Dataset hoặc kho dự án để giữ lại sau khi session kết thúc.

## Bước tiếp theo: C1 prompt ablation trên model nhỏ

Chạy [KAGGLE_C1_PROMPT_ABLATION.md](KAGGLE_C1_PROMPT_ABLATION.md). Cell nhận `PROMPT_VARIANT` là `original` hoặc `compact`; chạy hai Kaggle session song song với mọi cấu hình/seed giống nhau, chỉ khác prompt. Cell tạo validation stratified từ train, không đọc test chính thức; dùng Qwen2.5-Coder-1.5B-Instruct QLoRA 4-bit và constrained scoring 8 nhãn để metric hai prompt cùng protocol. So sánh token, thời gian train, Accuracy, Macro-F1 và F1 từng lớp trên validation. Chỉ sau khi chọn prompt mới tiến hành test ở ngày 6.
