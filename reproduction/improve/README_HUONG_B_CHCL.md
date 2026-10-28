# Hướng B: Tăng độ tin cậy phân loại commit (CHCL)

## Đọc nhanh

Tài liệu này chuyển hướng nghiên cứu sang phân loại loại commit (CHCL). Baseline hiện có là Llama-2-7B + QLoRA 4-bit, Run 1 đạt Macro-F1 0,841 và Accuracy 0,776 trên 518 mẫu test chính thức. Mục tiêu của hướng B không chỉ là tăng F1, mà còn làm dự đoán đáng tin cậy hơn: chỉ trả về một trong tám nhãn, có điểm tin cậy, đo calibration và đánh giá bằng nhiều seed.

**Lưu ý:** các mục tiêu và mức cải thiện bên dưới là giả thuyết, chưa phải kết quả. Kết quả của mỗi cell Kaggle cần được gửi lại và ghi vào README sau khi chạy.

## 1. Mục tiêu và giả thuyết

### Mục tiêu chính

Cải thiện độ tin cậy của phân loại commit bằng model nhỏ, dùng Llama-2-7B QLoRA làm baseline tham chiếu và giữ test chính thức 518 mẫu để so sánh với reproduction.

### Giả thuyết cần kiểm chứng

- F1 của `fix` và `refactor` tăng. Baseline lần lượt đạt khoảng 0,71 và 0,73; 49/116 lỗi thuộc hai hướng nhầm `fix → refactor` và `refactor → fix`.
- Chấm điểm trực tiếp trên tám nhãn loại bỏ invalid output do sinh tự do. Điểm xác suất và Expected Calibration Error (ECE) được báo cáo riêng.
- Bỏ phần định nghĩa nhãn lặp lại trong template có thể rút ngắn input đáng kể. Mức giảm khoảng một nửa là giả thuyết cần đo, không suy ra từ audit CHSUM: prompt CHCL và prompt CHSUM là hai template khác nhau.
- Kết luận chính có ít nhất ba seed hoặc 5-fold, khoảng tin cậy 95%, kiểm định thống kê phù hợp và báo cáo metric từng lớp.

### Đóng góp dự kiến

1. Audit reproduction: lỗi right-truncation làm mất nhãn, Run 2 suy giảm ở lớp `feat`, và protocol cổng chất lượng trước khi báo cáo kết quả.
2. Audit đầu vào: prompt CHCL gốc so với prompt ngắn; summary-only, diff-only và summary + diff.
3. Decoding ràng buộc trên tám nhãn bằng log-likelihood thay cho sinh nhãn tự do; đo invalid, xác suất và calibration.
4. Phân tích nhầm lẫn `fix`/`refactor`: loss có trọng số hoặc focal loss; thử `subtype` như tín hiệu phụ/phân cấp; audit nhãn bất đồng.
5. So sánh chi phí và chất lượng với baseline encoder và code-SLM nhỏ.

## 2. Các nhánh thí nghiệm

| Mã  | Câu hỏi / thay đổi               | Cách đo                                                                     |
| :-- | :------------------------------- | :-------------------------------------------------------------------------- |
| C1  | Bỏ template CHCL cố định         | Prompt gốc và prompt rút gọn trên cùng model/split; so F1, token, thời gian |
| C2  | Xác định nguồn thông tin hữu ích | Summary-only, diff-only, summary + diff                                     |
| C3  | Chấm nhãn trong tập đóng         | Log-likelihood của tám nhãn; Accuracy, Macro-F1, invalid, ECE               |
| C4  | Giảm nhầm `fix`/`refactor`       | Weighted CE hoặc focal loss; ablation `subtype`; audit nhãn thủ công        |
| C5  | Tăng độ chắc chắn của kết luận   | ≥3 seed hoặc 5-fold, bootstrap CI 95%, kiểm định cặp, báo cáo theo lớp      |
| C6  | Baseline và chi phí              | UniXcoder/CodeBERT và code-SLM 1.5–3B; VRAM, thời gian train/inference      |

Quy tắc tài nguyên: C1/C2/C4 chạy ablation trên một model nhỏ. Dùng 7B một lần cho cấu hình cuối để so với baseline paper; một lượt 7B trước đây mất khoảng 6,7 giờ GPU.

## 3. Kế hoạch

### Kế hoạch hoàn thành trong 7 ngày

Mục tiêu là hoàn tất **toàn bộ thí nghiệm cải tiến CHCL trong một tuần**, không chỉ pilot. Hãy dùng nhiều tài khoản/session Kaggle song song; mỗi session tối đa 12 giờ. Khi dữ liệu và cấu hình đã khóa, chạy C1/C2, C4, C6 và các seed ở những session độc lập. Mỗi session phải lưu checkpoint, predictions, logs và manifest ra Kaggle Output; không giả định `/kaggle/working` được chia sẻ giữa các session.

| Ngày | Việc                                                                                                                                                                                                                                    | Đầu ra mong đợi                                                                                                       |
| :--- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| 1    | Khóa dữ liệu/split/hash và revision; hoàn thiện harness, metric 8 lớp, bootstrap, ECE và cổng kiểm tra. Khởi chạy các session song song: adapter Run 1/C3, chuẩn bị SLM, encoder và annotation.                                         | Harness + manifest; các session có log/checkpoint riêng. Nếu thiếu adapter, bắt đầu train lại Run 1 trong session 7B. |
| 2    | Chạy C3 trên adapter Run 1. Trên SLM, chạy baseline CE và C1 prompt gốc/rút gọn; đo độ dài, thời gian và VRAM.                                                                                                                          | C3 metrics; C1 comparison; pilot qua cổng chất lượng.                                                                 |
| 3    | Chạy đầy đủ C2: summary-only, diff-only, summary + diff; song song tiếp tục các job C1. Chọn cấu hình theo validation, không xem test để chọn.                                                                                          | Ablation C1/C2 và tối đa hai cấu hình ứng viên.                                                                       |
| 4    | Chạy song song C4: CE, weighted CE, focal loss và `subtype` auxiliary trên cùng SLM/split. Khởi chạy UniXcoder, CodeBERT và code-SLM làm C6 ở các session riêng.                                                                        | Bảng F1 theo lớp, nhất là `fix`/`refactor`; chi phí từng baseline; giữ nhánh qua cổng chất lượng.                     |
| 5    | Chạy song song seed 0/1/2 cho baseline và cấu hình cải tiến tốt nhất trên validation. Hai người gán nhãn độc lập 100–200 lỗi/bất đồng `fix`/`refactor`.                                                                                 | Tối thiểu ba seed cho hai cấu hình chính; hai bộ nhãn độc lập đã hoàn tất.                                            |
| 6    | Chốt đối chiếu nhãn và Cohen's κ; chạy test chính thức một lần cho cấu hình đã chốt; bootstrap CI 95%, McNemar trên dự đoán cặp, confusion theo subtype; kiểm tra ví dụ lỗi. Hoàn tất rà soát tài liệu có hệ thống ở mức scope/novelty. | Kết quả test và thống kê; κ; bảng evidence, chi phí, giới hạn và nguồn tài liệu.                                      |
| 7    | Đối chiếu artifact/metadata, tổng hợp báo cáo và slide 8–10 trang; nêu cấu hình nào được xác nhận, cấu hình nào mới là pilot, quyết định hướng bài.                                                                                     | Bộ kết quả tái lập được và báo cáo cho cô; không biến giả thuyết thành kết luận.                                      |

### Tổ chức nhiều session Kaggle

- Mỗi session tối đa 12 giờ. Gán một job/configuration cho mỗi session, đặt tên thư mục output riêng và lưu command, seed, split hash, model/tokenizer revision, dtype, quantization, log, predictions, metrics và checkpoint.
- Upload artifact cần giữ lại thành Kaggle Dataset hoặc tải về trước khi session kết thúc; không dựa vào việc các tài khoản/session nhìn thấy cùng `/kaggle/working`.
- Luồng 7B chỉ chạy C3 inference trên adapter Run 1. Nếu adapter không có, dùng một session riêng để train lại Run 1 theo đúng cấu hình đã xác minh rồi mới chạy C3; không dùng Run 2 thay thế.
- Chạy selection trên validation. Test 518 mẫu chỉ mở một lần cho các cấu hình chốt trước; không tinh chỉnh dựa trên test.
- Nếu một run không qua cổng (loss không hữu hạn, output rỗng/lặp, lớp không được dự đoán), dừng nhánh đó và không nhân bản sang nhiều seed.

### Phạm vi và tính thực tế

Kế hoạch này đặt mục tiêu chạy tất cả nhánh C1–C6 trong tuần nhờ song song hóa; nếu quota/session hoặc cài đặt không ổn định, ưu tiên theo thứ tự: harness/C3, C1+C2, C4, ba seed cho baseline và cấu hình tốt nhất, thống kê, rồi mới thêm code-SLM/subtype. Ba seed là mức xác nhận tối thiểu cho cấu hình chính; các ablation phụ có thể chỉ có một seed và phải được gọi là thăm dò. Hoàn thành thí nghiệm và báo cáo trong tuần không đồng nghĩa bài báo đã sẵn sàng nộp hoặc Q3 được bảo đảm.

## 4. Cell Kaggle: Ngày 1, C3 trên adapter Run 1

Cell dưới đây dùng dataset CHCL gốc (`train.json`, `test.json`) và adapter Run 1 được attach như Kaggle Input. Nó chấm điểm tám nhãn bằng tổng log-likelihood của chuỗi nhãn và EOS. Vì chỉ chọn từ tám nhãn hợp lệ, invalid bằng 0 theo thiết kế; đây không phải bằng chứng rằng model tự do không thể sinh invalid.

ECE trong cell là **ECE sơ bộ của softmax trên tám điểm log-likelihood**, không phải xác suất đã hiệu chỉnh bằng temperature scaling. Không fit temperature trên test. Nếu adapter Run 1 không còn, cell dừng: cần train lại đúng Run 1 hoặc cung cấp adapter lưu trữ; không thay bằng adapter Run 2.

### Chuẩn bị Kaggle

1. Attach dataset gốc HQCM có `train.json` và `test.json`.
2. Attach adapter Run 1 dưới dạng Kaggle Dataset và đặt `ADAPTER_DIR` tới thư mục chứa `adapter_config.json` cùng `adapter_model.safetensors`.
3. Chấp nhận điều khoản Llama 2 trên Hugging Face và lưu Kaggle Secret `HF_TOKEN`.
4. Bật GPU. Cell cần tải base model 7B lượng tử hóa 4-bit; adapter không chứa base weights.
5. Nếu kernel chưa có thư viện phù hợp, cài dependencies trong cell riêng rồi restart kernel:

```python
%pip install -q transformers==4.46.1 peft==0.13.2 bitsandbytes scikit-learn pandas
```

### Cell

````python
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DATA_DIR = None  # Example: Path("/kaggle/input/hqcm/dataset")
ADAPTER_DIR = None  # Example: Path("/kaggle/input/hqcm-run1/adapter")
OUTPUT_DIR = Path("/kaggle/working/hqcm_huong_b_c3")
MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_REVISION = None  # Set a pinned Hugging Face commit SHA for repeat runs.
MAX_INPUT_TOKENS = 512
BATCH_SIZE = 8
BOOTSTRAP_RESAMPLES = 2000
ECE_BINS = 10
SEED = 42
LABELS = ["style", "docs", "test", "build", "cicd", "fix", "feat", "refactor"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if DATA_DIR is None:
    data_dirs = sorted({
        path.parent
        for path in Path("/kaggle/input").rglob("train.json")
        if (path.parent / "test.json").is_file()
    })
    if len(data_dirs) != 1:
        raise RuntimeError(
            "Set DATA_DIR to the folder containing the intended raw train.json and test.json."
        )
    DATA_DIR = data_dirs[0]
else:
    DATA_DIR = Path(DATA_DIR)

if ADAPTER_DIR is None:
    raise RuntimeError(
        "Set ADAPTER_DIR to the uploaded Run 1 adapter. "
        "If it is unavailable, retrain the verified Run 1 adapter first; do not use Run 2."
    )
ADAPTER_DIR = Path(ADAPTER_DIR)
if not (ADAPTER_DIR / "adapter_config.json").is_file():
    raise FileNotFoundError(f"Adapter config not found under {ADAPTER_DIR}")
if not (ADAPTER_DIR / "adapter_model.safetensors").is_file():
    raise FileNotFoundError(f"Adapter weights not found under {ADAPTER_DIR}")
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before loading the 7B model.")

train_path = DATA_DIR / "train.json"
test_path = DATA_DIR / "test.json"
if not train_path.is_file() or not test_path.is_file():
    raise FileNotFoundError(f"Expected raw train.json and test.json under {DATA_DIR}")

with train_path.open(encoding="utf-8") as file:
    train_rows = json.load(file)
with test_path.open(encoding="utf-8") as file:
    test_rows = json.load(file)
if len(train_rows) != 4611 or len(test_rows) != 518:
    raise ValueError(
        f"Expected official split 4611/518; got {len(train_rows)}/{len(test_rows)}."
    )

CHCL_PROMPT_TEMPLATE = """\
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

{diff}

```

Accompanying this code diff is its commit message:

```

{message}

```

Given this information, the git commit can be categorized as type: """

if any(row.get("type") not in LABELS for row in train_rows + test_rows):
    raise ValueError("The raw dataset contains labels outside the fixed eight-label set.")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient

        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as error:
        raise RuntimeError("Add the Hugging Face token as Kaggle Secret HF_TOKEN.") from error

tokenizer_kwargs = {"token": hf_token}
if MODEL_REVISION:
    tokenizer_kwargs["revision"] = MODEL_REVISION
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
tokenizer.truncation_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token_id is None:
    raise ValueError("The tokenizer must define an EOS token for candidate scoring.")

resolved_revision = tokenizer.init_kwargs.get("_commit_hash")
if not resolved_revision:
    raise RuntimeError("Could not resolve the base-model commit SHA; pin MODEL_REVISION.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=hf_token,
    revision=resolved_revision,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()
input_device = model.get_input_embeddings().weight.device


def build_prompt(row):
    return CHCL_PROMPT_TEMPLATE.format(
        diff=row["change"],
        message=row["summaries"]["en"],
    )


def candidate_token_ids(label):
    label_ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    if not label_ids:
        raise ValueError(f"Empty tokenization for label: {label}")
    return label_ids + [tokenizer.eos_token_id]


label_token_ids = {label: candidate_token_ids(label) for label in LABELS}


def score_candidates(prompt):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )["input_ids"]
    sequences = [
        prompt_ids + label_token_ids[label]
        for label in LABELS
    ]
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

    scores = []
    prompt_length = len(prompt_ids)
    for row_index, label in enumerate(LABELS):
        target_ids = label_token_ids[label]
        target_positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(target_ids) - 1,
            device=input_device,
        )
        target_tokens = torch.tensor(target_ids, dtype=torch.long, device=input_device)
        token_scores = log_probabilities[row_index, target_positions, target_tokens]
        scores.append(token_scores.sum())

    return torch.stack(scores).cpu()


gold_labels = [str(row["type"]).strip().lower() for row in test_rows]
score_rows = []
predictions = []
probabilities = []

for index, row in enumerate(test_rows, start=1):
    scores = score_candidates(build_prompt(row))
    class_probabilities = torch.softmax(scores, dim=0).numpy()
    prediction = LABELS[int(class_probabilities.argmax())]
    predictions.append(prediction)
    probabilities.append(class_probabilities)
    score_rows.append({
        "index": index - 1,
        "gold_label": gold_labels[index - 1],
        "predicted_label": prediction,
        "confidence": float(class_probabilities.max()),
        **{f"prob_{label}": float(class_probabilities[label_index])
           for label_index, label in enumerate(LABELS)},
    })
    if index % 50 == 0 or index == len(test_rows):
        print(f"Scored {index}/{len(test_rows)}")

assert len(predictions) == len(test_rows)
assert set(predictions).issubset(set(LABELS))
probability_array = np.asarray(probabilities, dtype=np.float64)
confidence = probability_array.max(axis=1)
correct = np.asarray(predictions) == np.asarray(gold_labels)


def expected_calibration_error(confidence_values, correct_values, n_bins=ECE_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            in_bin = (confidence_values >= lower) & (confidence_values <= upper)
        else:
            in_bin = (confidence_values >= lower) & (confidence_values < upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(
                correct_values[in_bin].mean() - confidence_values[in_bin].mean()
            )
    return float(ece)


def calculate_metrics(indices):
    selected_gold = np.asarray(gold_labels)[indices]
    selected_predictions = np.asarray(predictions)[indices]
    return {
        "accuracy": float(accuracy_score(selected_gold, selected_predictions)),
        "macro_f1_8": float(f1_score(
            selected_gold,
            selected_predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )),
    }


all_indices = np.arange(len(test_rows))
point_metrics = calculate_metrics(all_indices)
rng = np.random.default_rng(SEED)
bootstrap_values = {name: [] for name in point_metrics}
for _ in range(BOOTSTRAP_RESAMPLES):
    sample_indices = rng.integers(0, len(test_rows), size=len(test_rows))
    sample_metrics = calculate_metrics(sample_indices)
    for metric_name, metric_value in sample_metrics.items():
        bootstrap_values[metric_name].append(metric_value)

metrics = {
    **point_metrics,
    "macro_precision_8": float(precision_score(
        gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
    )),
    "macro_recall_8": float(recall_score(
        gold_labels, predictions, labels=LABELS, average="macro", zero_division=0
    )),
    "micro_f1_8": float(f1_score(
        gold_labels, predictions, labels=LABELS, average="micro", zero_division=0
    )),
    "invalid_count": 0,
    "invalid_rate": 0.0,
    "ece_10_bins_raw": expected_calibration_error(confidence, correct),
    "ece_note": "Uncalibrated ECE of softmax over summed label-plus-EOS log-likelihoods; exploratory only.",
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": SEED,
    "n_samples": len(test_rows),
    "labels": LABELS,
    "base_model": MODEL_ID,
    "base_model_revision": resolved_revision,
    "adapter_dir": str(ADAPTER_DIR),
    "max_input_tokens_left_truncated": MAX_INPUT_TOKENS,
    "scoring": "summed conditional token log-likelihood of label plus EOS",
}
for metric_name, values in bootstrap_values.items():
    metrics[f"{metric_name}_bootstrap_95ci"] = [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]

print(classification_report(
    gold_labels,
    predictions,
    labels=LABELS,
    target_names=LABELS,
    zero_division=0,
    digits=4,
))
print(json.dumps(metrics, ensure_ascii=False, indent=2))
pd.DataFrame(score_rows).to_csv(OUTPUT_DIR / "c3_test_scores.csv", index=False)
with (OUTPUT_DIR / "c3_metrics.json").open("w", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
print("Saved C3 artifacts to:", OUTPUT_DIR)
````

### Cách đọc kết quả và điều kiện dừng

- `invalid_count` bằng 0 vì dự đoán bị giới hạn tám nhãn. Đây là tính chất của phương pháp decode, không phải kết quả model tự sinh tự do.
- `ece_10_bins_raw` là ECE thăm dò từ softmax các log-likelihood; chưa temperature-scale. Không điều chỉnh calibration trên test. Nếu làm calibration chính thức, cần tập validation không tham gia huấn luyện adapter.
- So `macro_f1_8`, `accuracy`, CI và F1 từng lớp với Run 1. C3 chỉ được coi là có ích nếu cải thiện độ tin cậy mà không làm suy giảm đáng kể chất lượng.
- Lưu `c3_metrics.json`, `c3_test_scores.csv`, commit SHA model và adapter cùng báo cáo. Không dùng Run 2 làm baseline.
- Nếu không có adapter Run 1, dừng ở bước kiểm tra đường dẫn và khôi phục/train lại adapter Run 1 đã xác minh trước khi đánh giá.

## 5. Rủi ro và tiêu chí quyết định Q3

- Không thể bảo đảm tạp chí Q3 trước kết quả phản biện; kiểm tra danh mục Scopus/SCImago tại thời điểm nộp.
- Test chính thức chỉ có 11 mẫu cho mỗi lớp `build`, `cicd`, `style`; macro-F1 từng lớp có phương sai cao. Cần nhiều seed/fold và CI, không kết luận từ một lần chạy.
- Giá trị nghiên cứu nằm ở protocol có thể tái lập, audit lỗi/nhãn và phân tích calibration/chi phí, không chỉ ở một con số F1.
- Phạm vi lõi: C1 + C2 + C3 + C4 + C5. C6 là phần bổ sung nếu đủ thời gian.

## 6. Báo cáo sau mỗi lần chạy

Sau khi chạy cell Kaggle, ghi lại vào README_v4 (hoặc bản kế hoạch đang dùng): cấu hình, model revision, adapter revision/hash, seed, split, Accuracy, Macro-F1 (8 nhãn), F1 từng lớp, ECE, CI, thời gian và VRAM. Nêu rõ mọi lỗi hoặc kiểm tra cổng chưa đạt; không biến giả thuyết thành kết quả.
