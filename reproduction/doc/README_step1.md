# Bước 1 — Tái hiện (reproduction) HQCM / tác vụ Change Classification (CHCL)

Paper: *A Small-Scale yet High-Quality Dataset for Code Change Understanding* (HQCM) — Li et al., **"Understanding Code Changes Practically with Small-Scale Language Models"**, ASE '24.
Repo gốc: `codefuse-ai/codefuse-hqcm`, commit `ea7e7fab1bca176d9afd77350496649d7ca4743f`.

Notebook liên quan:
- [codefuse123.ipynb](../codefuse123.ipynb) — lần chạy thứ nhất (bị lỗi nhãn) + baseline TF-IDF.
- [codefuse123 (1).ipynb](../codefuse123%20(1).ipynb) — chẩn đoán lỗi, sửa preprocessing, train lại (`fixed_v2`) và evaluation chính thức.
- Kết quả: [hqcm_chcl_predictions.json](../hqcm_chcl_predictions.json), [baseline_manifest.json](../hqcm_baseline_audit/hqcm_baseline_audit/baseline_manifest.json), [classification_report.json](../hqcm_baseline_audit/hqcm_baseline_audit/classification_report.json), [confusion_matrix.csv](../hqcm_baseline_audit/hqcm_baseline_audit/confusion_matrix.csv).

---

## 1. Môi trường

| Thành phần | Giá trị |
|---|---|
| Nền tảng | Kaggle Notebook, 1 GPU (FP16, không hỗ trợ BF16) |
| Python / Torch | 3.13 / 2.11.0+cu128 |
| transformers / peft / bitsandbytes | 4.46.1 / (cập nhật `-U`) / (cập nhật `-U`) |
| scikit-learn | 1.6.1 |
| Base model | `meta-llama/Llama-2-7b-hf` (HF token qua Kaggle Secret) |

## 2. Dữ liệu

- Sinh biến thể CHCL bằng script chính thức: `python -m hqcm.xdata --task chcl --output /kaggle/working/hqcm_chcl ./dataset/`
- Train: **4,611** mẫu — Test: **518** mẫu, mỗi mẫu gồm `prompt` (mô tả 8 loại + diff + commit message) và `answer` (nhãn).
- Phân bố test: refactor 148, fix 133, feat 114, test 50, docs 40, build 11, cicd 11, style 11.

## 3. Các chỉnh sửa so với repo gốc

| # | File | Thay đổi | Lý do |
|---|---|---|---|
| 1 | `hqcm/xdata.py` | `args.fewshot` → `args.few_shot` | Bug của repo, script crash |
| 2 | `hqcm/sft.py` | Đọc dataset JSON local (`load_dataset("json", ...)`) | Chạy offline trên Kaggle |
| 3 | `hqcm/sft.py` | `bf16` → `fp16`, `bnb_4bit_compute_dtype=float16` | GPU Kaggle không hỗ trợ BF16 |
| 4 | `hqcm/sft.py` | `num_proc ≤ 4`, `save_steps = 25` | Giới hạn tài nguyên, resume khi hết phiên |
| 5 | `hqcm/sft.py` | **Viết lại `tokenize_and_preprocess`** (xem mục 4) | Sửa lỗi mất nhãn khi truncation |

## 4. Lần chạy 1 — thất bại và nguyên nhân

- Cấu hình đúng README gốc: `lr=2e-4, epochs=5, batch=1, LoRA r=64/α=16/dropout=0.1, max_length=512, seed=0, quantization=-1`.
- Train 23,055 step (~5h39m), `train_loss = 0.531`.
- Evaluation: **Accuracy 0.19%**, 516/518 output không phải nhãn (model sinh tiếp nội dung diff).
- **Nguyên nhân gốc:** Prompt CHCL phần lớn dài > 512 token (vd. 659–782 token). Hàm gốc tokenize `prompt + answer` rồi cắt phải (right-truncation) ⇒ phần câu hỏi cuối prompt và **nhãn bị cắt mất**; đồng thời mask theo độ dài prompt đã bị cắt ⇒ loss thực tế được tính trên token diff, model học "viết tiếp diff" thay vì phân loại.

## 5. Lần chạy 2 (`fixed_v2`) — kết quả chính

**Sửa preprocessing:** giữ `answer + EOS` trước, cắt **trái** prompt (giữ phần cuối gồm diff cuối + commit message + câu hỏi), chỉ tính loss trên `answer + EOS`. Đã kiểm tra 5 mẫu: target decode == answer (PASS).

**Cấu hình train:** như lần 1 nhưng `-Q 4` (QLoRA 4-bit NF4, double-quant, compute FP16).

**Inference:** script tự viết (không dùng nguyên `hqcm/gen.py`), base 4-bit + LoRA adapter, `truncation_side="left"`, `max_input=512`, greedy (`do_sample=False`), `max_new_tokens=16`.

| Chỉ số (518 mẫu test) | Giá trị |
|---|---|
| Accuracy | **0.7761** (402/518) |
| Macro-F1 (8 lớp, output lỗi tính là sai) | **0.8410** |
| Weighted-F1 | 0.7776 |
| Output không hợp lệ | 1/518 (model sinh `rename`, nhãn thật `refactor`) |

Per-class F1: build 0.917 · cicd 0.952 · docs 0.883 · feat 0.809 · fix 0.706 · refactor 0.731 · style 0.870 · test 0.860.

**Phân tích lỗi (confusion matrix):** 116 lỗi, trong đó **88 lỗi (~76%)** nằm trong cụm `fix ↔ refactor ↔ feat` (fix→refactor 26, refactor→fix 23, feat→fix 14, fix→feat 11, feat→refactor 8, refactor→feat 6). Các lớp hiếm (build/cicd/style) lại dễ.

## 6. Baseline cổ điển (cùng train/test)

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF (1–2gram) + Logistic Regression (balanced) | 0.5792 | 0.6847 | 0.5778 |
| TF-IDF (1–2gram) + Linear SVM (balanced) | 0.6255 | 0.7100 | 0.6240 |
| TF-IDF (1–3gram, min_df=2, C=1.0, tune trên 15% validation) + Linear SVM | 0.6313 | 0.7261 | — |
| **Llama-2-7B + LoRA (HQCM, fixed_v2)** | **0.7761** | **0.8410** | **0.7776** |

LLM fine-tune vượt baseline tốt nhất **+14.5 điểm Accuracy** và **+11.5 điểm Macro-F1**.

## 6b. Đối chiếu với paper (ASE '24, Table 4 — HQCM-finetuned Llama2-7b)

Nguồn: [3691620.3694999.pdf](3691620.3694999.pdf), mục 4.3 / Table 4. Tập test **trùng khớp**: support từng lớp suy ra từ recall trong paper (vd. style 8/11, docs 33/40, feat 86/114, refactor 98/148, fix 105/133) đúng bằng phân bố 518 mẫu của ta; micro 75.87% = 393/518.

Lưu ý định nghĩa: Macro-F1 của paper = $\frac{2PR}{P+R}$ với $P, R$ là macro-precision/recall (80.97 = HM(80.60, 81.34)), **không phải** trung bình F1 từng lớp. Micro-F1 = Accuracy (bài toán đơn nhãn).

| Chỉ số (%) | Paper | Reproduction | Δ |
|---|---|---|---|
| Micro-F1 (= Accuracy) | 75.87 | **77.61** | +1.74 |
| Macro-Precision | 80.60 | **83.94** | +3.34 |
| Macro-Recall | 81.34 | **84.58** | +3.24 |
| Macro-F1 (định nghĩa paper, HM của macro P/R) | 80.97 | **84.26** | +3.29 |
| Macro-F1 (trung bình F1 từng lớp) | 80.72* | **84.10** | +3.38 |

\* tự tính từ F1 từng lớp trong Table 4.

| F1 từng lớp (%) | Paper | Reproduction | Δ |
|---|---|---|---|
| feat | 80.75 | 80.91 | +0.16 |
| refactor | 69.50 | 73.09 | +3.59 |
| style | 72.73 | 86.96 | +14.23 (11 mẫu, ±1 mẫu ≈ ±9 điểm) |
| docs | 83.54 | 88.31 | +4.77 |
| build | 91.67 | 91.67 | 0.00 |
| fix | 70.95 | 70.63 | −0.32 |
| test | 85.71 | 86.00 | +0.29 |
| cicd | 90.91 | 95.24 | +4.33 |

**Nhận định:** Kết quả tái hiện **đạt và nhỉnh hơn paper** (+1.7 điểm micro, +3.3 điểm macro). Mức chênh micro (+9 mẫu đúng) nằm trong khoảng tin cậy 95% (~±3.6 điểm) nên xem là **tái hiện thành công**. Phần chênh macro chủ yếu do các lớp rất nhỏ (style, cicd: 11 mẫu) — một mẫu đổi kết quả đã dịch F1 rất nhiều. Các lớp lớn (feat, fix, test, build) gần như trùng khớp. Paper không công bố chi tiết truncation; việc giữ phần cuối prompt (commit message + câu hỏi) nhiều khả năng là lý do kết quả không thấp hơn dù dùng QLoRA 4-bit.

Ghi chú: paper viết "80% train / 20% test" nhưng số liệu Table 4 khớp chính xác test 518 mẫu (≈10%) của repo, nên phép so sánh là cùng tập test.

---

## 7. Đánh giá: đã đủ điều kiện chuyển sang cải tiến chưa?

**Kết luận (cập nhật): ĐỦ ĐIỀU KIỆN chuyển sang cải tiến.** Pipeline CHCL chạy end-to-end, kết quả đạt/nhỉnh hơn paper (mục 6b), invalid 0.19%, vượt rõ baseline cổ điển. Ba mục bắt buộc bên dưới đã được xử lý; `fixed_v2` là **baseline chính thức** cho giai đoạn cải tiến.

### Đạt
- [x] Chạy được pipeline chính thức (xdata → sft → inference) trên dữ liệu HQCM gốc, test set cố định 518 mẫu.
- [x] Chẩn đoán và sửa được lỗi truncation nghiêm trọng của `sft.py` (có kiểm chứng).
- [x] Có baseline cổ điển + tuning trên validation tách từ train (không leak test).
- [x] Lưu artifact: adapter, predictions, manifest (commit, version, hash, config), confusion matrix.

### Bắt buộc trước khi cải tiến (đã xử lý)
1. [x] **Đối chiếu số liệu với paper** — xem mục 6b: micro +1.74, macro +3.29 điểm so với Table 4.
2. [x] **Thống nhất cách tính Macro-F1.** Giá trị cũ `0.7476` chia trung bình cho 9 lớp (thêm lớp giả `INVALID`). Đã sửa [baseline_manifest.json](../hqcm_baseline_audit/hqcm_baseline_audit/baseline_manifest.json) và [classification_report.json](../hqcm_baseline_audit/hqcm_baseline_audit/classification_report.json): chỉ tính trên 8 lớp, output lỗi tính là sai; ghi cả 2 định nghĩa `macro_f1 = 0.8410` (trung bình F1) và `macro_f1_harmonic_of_macro_pr = 0.8426` (định nghĩa paper), thêm micro P/R/F1 và khối `paper_reference`.
3. [x] **Thống nhất file kết quả chính thức.** Nội dung [hqcm_chcl_predictions.json](../hqcm_chcl_predictions.json) **giống hệt** bản trên Kaggle; chỉ khác ký tự xuống dòng (CRLF do Windows/OneDrive). Sau khi chuẩn hóa CRLF→LF, hash = `4b6a9a27…` khớp manifest. File local là bản chính thức; manifest ghi cả 2 hash (`predictions_sha256_note`). Khi kiểm tra hash, luôn chuẩn hóa về LF trước.
   - Bản `sft.py` đã sửa (đúng bản dùng train `fixed_v2`): [patches/sft_fixed_v2.py](../patches/sft_fixed_v2.py), mỗi chỗ sửa đánh dấu `PATCH n`.
   - Adapter: nằm trên Kaggle (`/kaggle/working/hqcm_llama2_7b_chcl_fixed_v2`). Đã thêm **Cell 13** cuối [codefuse123 (1).ipynb](../codefuse123%20(1).ipynb) để đóng gói adapter + `sft.py`/`xdata.py` đã sửa + predictions + train/test đã xử lý kèm `SHA256SUMS.txt` vào `hqcm_chcl_fixed_v2_release.zip`. **Cần chạy cell này trên Kaggle** rồi lưu ZIP thành Kaggle Dataset (hoặc tải về) — việc này chưa làm được từ máy local.

### Nên làm (tăng độ tin cậy)
4. Chạy thêm 2–3 seed (hoặc ít nhất bootstrap CI trên test) để biết độ dao động; với 518 mẫu, sai số Accuracy ~±3.6 điểm (95%).
5. Dọn notebook: gộp 2 file thành một notebook "sạch" chạy tuần tự (hiện có cell trùng, cell comment, cell lần chạy lỗi), kèm `pip freeze`.

## 8. Các điểm sai khác so với thiết lập gốc (cần ghi rõ khi báo cáo)

| Hạng mục | Paper / repo gốc | Reproduction |
|---|---|---|
| Precision | BF16, không lượng tử hóa (`-Q -1`) | FP16, **QLoRA 4-bit** (`-Q 4`) |
| Preprocessing | Right-truncation `prompt+answer` (mất nhãn khi > 512) | Left-truncation prompt, giữ nhãn |
| Inference | `hqcm/gen.py`, không cắt input | Script riêng, cắt trái 512 token, base 4-bit |
| Phiên bản thư viện | theo `requirements.txt` | torch 2.11, transformers 4.46.1, Python 3.13 |
| Số seed | — | 1 (seed 0) |

## 9. Gợi ý hướng cải tiến (Bước 2)

Dựa trên phân tích lỗi (76% lỗi ở cụm fix/refactor/feat) và giới hạn 512 token:
1. **Tăng ngữ cảnh / cắt diff thông minh:** max_length 1024–2048, hoặc nén diff (giữ hunk header, dòng `+/-`, bỏ context lines) thay vì cắt trái mù.
2. **Đổi backbone SLM mới hơn** (Qwen2.5-Coder-1.5B/7B, DeepSeek-Coder, CodeLlama) với cùng pipeline để so sánh công bằng.
3. **Ràng buộc decoding theo tập nhãn** (scoring 8 nhãn bằng log-prob) → loại bỏ output không hợp lệ, cho xác suất để calibration.
4. **Xử lý nhầm lẫn fix/refactor/feat:** thêm đặc trưng/hint từ commit message, loss có trọng số lớp, hoặc data augmentation cho các cặp dễ nhầm.
5. **Ensemble** LLM + TF-IDF SVM (SVM mạnh ở lớp hiếm) hoặc few-shot (biến thể `--few-shot` của `xdata.py`).

Mọi cải tiến cần so sánh trên **cùng test 518 mẫu**, cùng định nghĩa metric (mục 7.2), và báo cáo so với baseline `fixed_v2` (Acc 0.7761 / Macro-F1 0.8410).
