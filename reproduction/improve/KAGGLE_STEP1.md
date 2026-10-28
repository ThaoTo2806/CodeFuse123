# Kaggle - Bước 1: Đếm token bằng tokenizer thật

## Cell này làm gì?

Cell đọc `train.json` và `test.json` gốc của HQCM, sau đó đếm token bằng tokenizer Llama 2 thật cho hai dạng input:

- `diff_only`: chỉ có phần diff, không có template.
- `chsum_prompt`: diff được đặt trong template tạo commit message chính thức.

Kết quả cho biết chính xác bao nhiêu mẫu vượt 512 token, thay cho ước lượng theo số ký tự. Cell không đọc adapter, checkpoint hay thư mục `hqcm_reproduction_80_20`; không cần upload các artifact nặng đó lên Kaggle.

## Chuẩn bị trên Kaggle

1. Bật Internet nếu tải tokenizer từ Hugging Face. Cell này chỉ đếm token bằng CPU, không cần bật GPU.
2. Attach dataset gốc có `train.json` và `test.json`. Mỗi dòng cần có các trường `change` và `summaries.en`; các file có thể nằm trong thư mục con, ví dụ `/kaggle/input/hqcm/dataset/`.
3. Llama 2 là model gated. Hãy chấp nhận điều khoản trên Hugging Face và tạo Kaggle Secret có tên `HF_TOKEN`. Cách này chỉ tải tokenizer nhỏ, không tải weights của model.
4. Nếu đã upload model/tokenizer vào Kaggle Input, gán đường dẫn thư mục local vào `MODEL_DIR` trong cell. Nếu không, để `MODEL_DIR = None` để tải tokenizer từ Hugging Face.
5. Nếu kernel chưa có Transformers, chạy một cell cài đặt trước:

```python
%pip install -q transformers==4.46.1 pandas
```

## Cell audit

Chạy cell dưới đây sau khi attach dataset và cấu hình tokenizer. Nếu Kaggle Input có nhiều dataset HQCM, cell ưu tiên cặp train/test có tổng 5.129 mẫu; nếu vẫn có nhiều lựa chọn thì cell dừng để tránh chọn nhầm dữ liệu.

````python
from pathlib import Path
import hashlib
import json
import os

import pandas as pd
from transformers import AutoTokenizer

# Co the gan duong dan ro rang neu Kaggle Input co nhieu dataset.
DATA_DIR = None  # Vi du: Path("/kaggle/input/hqcm/dataset")
MODEL_DIR = None  # Vi du: Path("/kaggle/input/llama-2-7b-hf")

MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_REVISION_LOCK = None  # Lan sau co the dien commit hash in ra ben duoi.
MAX_LENGTH = 512
OUTPUT_DIR = Path("/kaggle/working/improve_step1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_raw_hqcm_pair(train_path, test_path):
    try:
        with train_path.open(encoding="utf-8") as file:
            train_sample = json.load(file)
        with test_path.open(encoding="utf-8") as file:
            test_sample = json.load(file)
        return (
            isinstance(train_sample, list)
            and isinstance(test_sample, list)
            and bool(train_sample)
            and bool(test_sample)
            and {"change", "summaries", "type"}.issubset(train_sample[0])
            and {"change", "summaries", "type"}.issubset(test_sample[0])
            and "en" in train_sample[0]["summaries"]
            and "en" in test_sample[0]["summaries"]
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


if DATA_DIR is not None:
    data_dirs = [Path(DATA_DIR)]
else:
    data_dirs = sorted({path.parent for path in Path("/kaggle/input").rglob("train.json")})

pairs = [
    (directory / "train.json", directory / "test.json")
    for directory in data_dirs
    if (directory / "train.json").is_file()
    and (directory / "test.json").is_file()
    and is_raw_hqcm_pair(directory / "train.json", directory / "test.json")
]

if not pairs:
    raise FileNotFoundError(
        "Khong tim thay cap train.json/test.json HQCM goc. "
        "Hay attach dataset co cac truong change, summaries.en, type "
        "hoac gan DATA_DIR den thu muc chua hai file nay."
    )

if len(pairs) > 1:
    official_size_pairs = [
        pair for pair in pairs
        if len(json.loads(pair[0].read_text(encoding="utf-8")))
        + len(json.loads(pair[1].read_text(encoding="utf-8"))) == 5129
    ]
    if len(official_size_pairs) == 1:
        pairs = official_size_pairs
    else:
        paths = [str(train_path.parent) for train_path, _ in pairs]
        raise RuntimeError(
            "Co nhieu bo du lieu HQCM co the dung. Gan DATA_DIR cho dung bo: "
            + ", ".join(paths)
        )

train_path, test_path = pairs[0]
with train_path.open(encoding="utf-8") as file:
    train_rows = json.load(file)
with test_path.open(encoding="utf-8") as file:
    test_rows = json.load(file)
print(f"Dataset: {train_path.parent}")
print(f"So mau: train={len(train_rows):,}, test={len(test_rows):,}")

# Dung dung template chsum trong hqcm/xdata.py, khong lay prompt CHCL hay artifact reproduction.
def make_chsum_prompt(diff):
    return f"Please generate a commit message for the following diff:\n\n```diff\n{diff}\n```\n"


hf_token = None
if MODEL_DIR is not None:
    model_source = str(Path(MODEL_DIR))
    if not Path(model_source).is_dir():
        raise FileNotFoundError(f"MODEL_DIR khong ton tai: {model_source}")
    tokenizer_kwargs = {"local_files_only": True}
else:
    model_source = MODEL_ID
    try:
        from kaggle_secrets import UserSecretsClient

        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Can HF_TOKEN trong Kaggle Secrets, hoac gan MODEL_DIR den tokenizer "
            "da upload vao Kaggle Input."
        )
    tokenizer_kwargs = {"token": hf_token}
    if MODEL_REVISION_LOCK:
        tokenizer_kwargs["revision"] = MODEL_REVISION_LOCK

tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
resolved_revision = tokenizer.init_kwargs.get("_commit_hash")
if MODEL_DIR is not None:
    tokenizer_file_names = {
        "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json",
    }
    fingerprint = hashlib.sha256()
    local_path = Path(MODEL_DIR)
    for file_path in sorted(
        path for path in local_path.rglob("*")
        if path.is_file() and path.name in tokenizer_file_names
    ):
        fingerprint.update(file_path.relative_to(local_path).as_posix().encode("utf-8"))
        fingerprint.update(file_path.read_bytes())
    tokenizer_fingerprint = fingerprint.hexdigest()
    resolved_revision = resolved_revision or "local-upload"
else:
    tokenizer_fingerprint = None

print("Tokenizer:", model_source)
print("Resolved model revision:", resolved_revision)
print("Ghi lai commit hash tren de khoa revision cho lan chay sau.")

summary_rows = []
sample_rows = []
for split_name, rows in (("train", train_rows), ("test", test_rows)):
    for index, row in enumerate(rows):
        diff = row["change"]
        diff_token_count = len(
            tokenizer(diff, add_special_tokens=True, truncation=False)["input_ids"]
        )
        chsum_prompt = make_chsum_prompt(diff)
        prompt_token_count = len(
            tokenizer(chsum_prompt, add_special_tokens=True, truncation=False)["input_ids"]
        )
        sample_rows.append({
            "split": split_name,
            "index": index,
            "diff_tokens": diff_token_count,
            "chsum_prompt_tokens": prompt_token_count,
        })

    split_samples = [item for item in sample_rows if item["split"] == split_name]
    for input_name, length_key in (
        ("diff_only", "diff_tokens"),
        ("chsum_prompt", "chsum_prompt_tokens"),
    ):
        lengths = pd.Series([item[length_key] for item in split_samples], dtype="int64")
        over_limit = int((lengths > MAX_LENGTH).sum())
        summary_rows.append({
            "split": split_name,
            "input_view": input_name,
            "samples": len(lengths),
            "max_length": MAX_LENGTH,
            "over_limit_count": over_limit,
            "over_limit_rate": over_limit / len(lengths),
            "mean_tokens": float(lengths.mean()),
            "median_tokens": float(lengths.median()),
            "p90_tokens": float(lengths.quantile(0.90)),
            "max_tokens": int(lengths.max()),
            "model_id": model_source,
            "model_revision": resolved_revision,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "prompt_length_summary.csv", index=False)
pd.DataFrame(sample_rows).to_csv(OUTPUT_DIR / "prompt_token_lengths.csv", index=False)

metadata = {
    "model_id": model_source,
    "model_revision": resolved_revision,
    "model_revision_lock_used": MODEL_REVISION_LOCK,
    "tokenizer_fingerprint_sha256": tokenizer_fingerprint,
    "dataset_dir": str(train_path.parent),
    "train_samples": len(train_rows),
    "test_samples": len(test_rows),
    "max_length": MAX_LENGTH,
    "views": {
        "diff_only": "Raw code diff with tokenizer special tokens.",
        "chsum_prompt": "Official HQCM CHSUM wrapper around the raw code diff.",
    },
}
with (OUTPUT_DIR / "tokenizer_audit_metadata.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

display(summary)
print("Da luu ket qua tai:", OUTPUT_DIR)
print("CSV tong hop:", OUTPUT_DIR / "prompt_length_summary.csv")
````

## Đọc kết quả

- `over_limit_rate` là tỉ lệ prompt vượt quá 512 token. `diff_only` và `chsum_prompt` được tính riêng để thấy chi phí của template.
- `prompt_length_summary.csv` là bảng tổng hợp cho train/test. `prompt_token_lengths.csv` lưu số token từng mẫu để truy vết các trường hợp dài.
- `tokenizer_audit_metadata.json` ghi model, revision, đường dẫn dataset và fingerprint tokenizer nếu dùng model local.
- Nếu dùng Hugging Face, copy `Resolved model revision` từ output vào `MODEL_REVISION_LOCK` khi chạy lại để tải đúng commit. Nếu tokenizer là bản local upload, metadata đánh dấu `local-upload` và lưu fingerprint SHA-256.

## Tải kết quả

Sau khi chạy xong, tải các file trong Kaggle Output hoặc tạo ZIP nhỏ bằng cell này:

```python
import shutil

zip_path = shutil.make_archive(
    "/kaggle/working/improve_step1_results",
    "zip",
    root_dir="/kaggle/working",
    base_dir="improve_step1",
)
print("ZIP kết quả:", zip_path)
```
