# Kaggle - Bước 2: Chuẩn bị 100 mẫu gán nhãn thủ công

## Mục đích

Cell này chọn 100 cặp `diff → summaries.en` từ dataset HQCM gốc để người đọc đánh giá xem summary có nói điều diff không hỗ trợ hay không. Cell không tải model, tokenizer hay adapter; chỉ cần dataset gốc trong Kaggle Input hoặc `/kaggle/working` và `pandas`.

Cell tạo hai nhóm, có thể kiểm tra lại bằng seed cố định:

- 50 mẫu có ít nhất một định danh kiểu camelCase/snake_case trong summary nhưng không tìm thấy nguyên dạng trong diff. Đây chỉ là tín hiệu để tìm mẫu, **không đồng nghĩa định danh đó là bịa**.
- 50 mẫu có lexical coverage thấp nhất trong số mẫu không được chọn ở nhóm đầu. Coverage là tỉ lệ từ nội dung trong summary xuất hiện nguyên dạng trong diff, dùng làm cách sàng lọc đơn giản chứ không phải phép đo factuality.

Hai nhóm được cố ý lấy theo tiêu chí, không lấy ngẫu nhiên đại diện toàn bộ dữ liệu. Vì vậy không dùng tỉ lệ nhãn của 100 mẫu này để ước lượng trực tiếp tỉ lệ bịa trong cả HQCM.

## Cách gán nhãn

Mở file CSV trong Kaggle Output bằng spreadsheet. Đọc `summary_en` cùng `diff`, rồi điền cột `human_label` bằng đúng một trong các giá trị sau:

| Giá trị               | Chọn khi                                                                                                                                 |
| :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| `bịa thật`            | Summary khẳng định chi tiết cụ thể mà diff không hỗ trợ và chi tiết đó không chỉ là kiến thức/thuật ngữ hợp lệ hoặc ngữ cảnh ngoài diff. |
| `khái niệm hợp lệ`    | Từ/định danh không hiện nguyên dạng trong diff nhưng là tên khái niệm, thư viện, chuẩn, hoặc thuật ngữ hợp lệ phù hợp với thay đổi.      |
| `ngữ cảnh ngoài diff` | Summary nêu bối cảnh hoặc động cơ có thể đúng nhưng không thể xác nhận chỉ từ diff.                                                      |
| `khác`                | Không thuộc các nhóm trên, không rõ, hoặc không thể đánh giá. Ghi lý do ngắn trong `annotator_note`.                                     |

Không gán `bịa thật` chỉ vì một định danh bị flag hoặc vì coverage thấp. Đánh giá toàn bộ ý của summary trong ngữ cảnh diff. Nếu có hai người gán nhãn, mỗi người nên làm độc lập trước khi đối chiếu.

## Cell Kaggle

Có thể đặt `DATA_DIR` thành thư mục chứa `train.json` và `test.json`. Để `None` thì cell tìm trong `/kaggle/input` và `/kaggle/working`, ưu tiên bộ có tổng 5.129 mẫu. Nếu tìm thấy nhiều bộ không phân biệt được, cell dừng và yêu cầu chỉ rõ thư mục.

```python
from pathlib import Path
import json
import random
import re

import pandas as pd

DATA_DIR = None  # Vi du: Path("/kaggle/input/hqcm/dataset")
OUTPUT_DIR = Path("/kaggle/working/improve_step2_manual_labeling")
RANDOM_SEED = 42
UNSUPPORTED_IDENTIFIER_SAMPLES = 50
LOW_COVERAGE_SAMPLES = 50
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*|"
    r"[A-Za-z]+[A-Z][A-Za-z0-9]*|"
    r"[A-Z]{2,}[A-Za-z0-9]*"
    r")(?![A-Za-z0-9])"
)
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "for", "from", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "them", "then", "there",
    "these", "this", "those", "to", "was", "were", "will", "with",
}
LABEL_VALUES = [
    "bịa thật",
    "khái niệm hợp lệ",
    "ngữ cảnh ngoài diff",
    "khác",
]


def load_raw_split(train_path, test_path):
    try:
        train_rows = json.loads(train_path.read_text(encoding="utf-8"))
        test_rows = json.loads(test_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(train_rows, list) or not isinstance(test_rows, list):
        return None
    if not train_rows or not test_rows:
        return None

    required_keys = {"change", "summaries", "type"}
    for rows in (train_rows, test_rows):
        first_row = rows[0]
        if not isinstance(first_row, dict) or not required_keys.issubset(first_row):
            return None
        if not isinstance(first_row.get("summaries"), dict):
            return None
        if "en" not in first_row["summaries"]:
            return None
    return train_rows, test_rows


if DATA_DIR is not None:
    candidate_dirs = [Path(DATA_DIR)]
else:
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    candidate_dirs = sorted({
        path.parent
        for root in search_roots
        if root.exists()
        for path in root.rglob("train.json")
    })

valid_datasets = []
for directory in candidate_dirs:
    train_path = directory / "train.json"
    test_path = directory / "test.json"
    if not train_path.is_file() or not test_path.is_file():
        continue
    loaded = load_raw_split(train_path, test_path)
    if loaded is not None:
        valid_datasets.append((directory, *loaded))

if not valid_datasets:
    raise FileNotFoundError(
        "Khong tim thay dataset HQCM goc co train.json/test.json va "
        "cac truong change, summaries.en, type. Hay gan DATA_DIR."
    )

if len(valid_datasets) > 1:
    official_size = [
        item for item in valid_datasets
        if len(item[1]) + len(item[2]) == 5129
    ]
    if len(official_size) == 1:
        valid_datasets = official_size
    else:
        found_dirs = [str(item[0]) for item in valid_datasets]
        raise RuntimeError(
            "Tim thay nhieu bo dataset HQCM. Hay gan DATA_DIR ro rang: "
            + ", ".join(found_dirs)
        )

dataset_dir, train_rows, test_rows = valid_datasets[0]
print(f"Dataset: {dataset_dir}")
print(f"Train: {len(train_rows):,} | Test: {len(test_rows):,}")


def unique_identifiers(text):
    found = IDENTIFIER_PATTERN.findall(str(text))
    return list(dict.fromkeys(found))


def content_terms(text):
    return [
        word.lower()
        for word in WORD_PATTERN.findall(str(text))
        if len(word) > 1 and word.lower() not in STOPWORDS
    ]


records = []
for split_name, rows in (("train", train_rows), ("test", test_rows)):
    for row_index, row in enumerate(rows):
        summary = str(row["summaries"].get("en", "")).strip()
        diff = str(row["change"])
        if not summary:
            continue

        diff_lower = diff.lower()
        summary_identifiers = unique_identifiers(summary)
        unsupported_identifiers = [
            identifier
            for identifier in summary_identifiers
            if identifier.lower() not in diff_lower
        ]

        summary_terms = content_terms(summary)
        diff_terms = {word.lower() for word in WORD_PATTERN.findall(diff)}
        coverage = (
            sum(term in diff_terms for term in summary_terms) / len(summary_terms)
            if summary_terms else 0.0
        )
        terms_not_found = list(dict.fromkeys(
            term for term in summary_terms if term not in diff_terms
        ))

        records.append({
            "sample_id": f"{split_name}:{row_index}",
            "split": split_name,
            "row_index": row_index,
            "type": row.get("type", ""),
            "subtype": row.get("subtype", ""),
            "summary_en": summary,
            "diff": diff,
            "unmatched_identifiers": unsupported_identifiers,
            "lexical_coverage": coverage,
            "summary_terms_not_found_in_diff": terms_not_found,
        })

unsupported_pool = [record for record in records if record["unmatched_identifiers"]]
if len(unsupported_pool) < UNSUPPORTED_IDENTIFIER_SAMPLES:
    raise ValueError(
        f"Chi co {len(unsupported_pool)} mau co identifier bi flag; "
        f"khong du {UNSUPPORTED_IDENTIFIER_SAMPLES} mau."
    )

random_generator = random.Random(RANDOM_SEED)
unsupported_selected = random_generator.sample(
    unsupported_pool,
    UNSUPPORTED_IDENTIFIER_SAMPLES,
)
selected_ids = {record["sample_id"] for record in unsupported_selected}

# Nhóm phủ từ thấp không có identifier bị flag, để hai nhóm dễ phân biệt.
low_coverage_pool = [
    record for record in records
    if not record["unmatched_identifiers"]
    and record["sample_id"] not in selected_ids
]
if len(low_coverage_pool) < LOW_COVERAGE_SAMPLES:
    raise ValueError(
        f"Chi co {len(low_coverage_pool)} mau khong trung co identifier bi flag; "
        f"khong du {LOW_COVERAGE_SAMPLES} mau cho nhom coverage thap."
    )

low_coverage_selected = sorted(
    low_coverage_pool,
    key=lambda record: (
        record["lexical_coverage"],
        record["split"],
        record["row_index"],
    ),
)[:LOW_COVERAGE_SAMPLES]

annotation_rows = []
for group_name, selected in (
    ("unsupported_identifier", unsupported_selected),
    ("low_lexical_coverage", low_coverage_selected),
):
    for record in selected:
        annotation_rows.append({
            "sample_id": record["sample_id"],
            "sampling_group": group_name,
            "split": record["split"],
            "row_index": record["row_index"],
            "type": record["type"],
            "subtype": record["subtype"],
            "summary_en": record["summary_en"],
            "diff": record["diff"],
            "unmatched_identifiers_heuristic": "; ".join(record["unmatched_identifiers"]),
            "lexical_coverage_heuristic": round(record["lexical_coverage"], 4),
            "summary_terms_not_found_in_diff": "; ".join(record["summary_terms_not_found_in_diff"]),
            "human_label": "",
            "annotator_note": "",
        })

annotation_df = pd.DataFrame(annotation_rows)
assert len(annotation_df) == 100
assert annotation_df["sample_id"].nunique() == 100

output_path = OUTPUT_DIR / "manual_labels_100.csv"
annotation_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(annotation_df.groupby("sampling_group").size())
print("CSV de dien nhan:", output_path)
print("Seed:", RANDOM_SEED)
print("Nhan hop le:", LABEL_VALUES)
```

Sau khi điền `human_label`, có thể lưu CSV và upload lại vào Kaggle để tổng hợp. Không cần gửi lại diff hoặc summary riêng; chúng đã nằm trong mỗi dòng của file.
