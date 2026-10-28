# Evidence-Aware Fine-Tuning for Faithful Code Change Summarization

> **Tên làm việc (có thể đổi).** Cụm _"Functionally Hard Negatives"_ chỉ giữ nếu kiểm tra thủ công cho thấy các ví dụ âm thật sự khác chức năng. Nếu không, đổi thành _Category-Filtered Hard Negatives_.

**Trạng thái (28/10/2026):** đã hoàn thành reproduction pipeline HQCM ở task phân loại và audit dữ liệu/tokenizer. **Chưa** có baseline cho task tạo câu mô tả. Mọi nội dung trong mục "Giả thuyết" vẫn là giả thuyết cần kiểm chứng.

---

## 1. Đã làm

### 1.1 Reproduction task phân loại (CHCL), Llama-2-7B + QLoRA 4-bit

| Run       | Split                            | Kết quả                                                    | Kết luận                                                                                                                                                             |
| :-------- | :------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Run 1** | 4.611 train / 518 test           | Macro-F1 **0,841** (8 nhãn), Accuracy **0,776**, 1 invalid | **Hợp lệ.** Nhất quán với paper (Macro-F1 0,810; Micro-F1 0,759). Dùng làm baseline pipeline                                                                         |
| **Run 2** | 4.103 train / 1.026 test (80/20) | Macro-F1 0,731, Accuracy 0,633, 0 invalid                  | **Hỏng theo lớp:** nhãn `feat` không được sinh lần nào trong 1.026 output (176/227 mẫu `feat` bị dự đoán là `fix`, 48 là `refactor`). Không dùng làm số liệu so sánh |

Phát hiện và bài học từ reproduction:

- Code gốc `sft.py` cắt phải prompt dài, làm **mất nhãn** khi huấn luyện (hai lần train đầu cho accuracy 0,19% và 3,3%). Đã sửa bằng preprocessing riêng: giữ nhãn + EOS, cắt từ đầu prompt, mask phần prompt.
- Cách tính Macro-F1: tính trên **8 nhãn thật**. Con số 0,7476 trong `manifest` là trung bình 9 lớp (có lớp `INVALID` support 0), không dùng.
- Run 2 dùng FP16 (log có `grad_norm: nan` ở bước 0) và train loss về ≈ 0 từ epoch 3. Nguyên nhân chính xác của lỗi `feat` **chưa xác định** (nghi: FP16, lệch tokenize train/inference, overfit).
- `model_revision` đang `null` trong metadata, cần khóa commit hash của model base.

### 1.2 Audit dữ liệu HQCM (5.129 mẫu; `hqcm_audit.py`)

**Cấu trúc.** `train.json`/`test.json` chỉ có `prompt` và `answer` (nhãn phân loại), không có file diff → tóm tắt riêng. Mỗi prompt gồm: template cố định 1.533 ký tự (định nghĩa 8 loại) + diff + commit message + câu hỏi cuối. Cặp (diff, message) phải tách từ prompt (không mẫu nào lỗi parse).

**Phân bố nhãn:** refactor 1.472 · fix 1.324 · feat 1.132 · test 500 · docs 397 · build 102 · style 101 · cicd 101.

**Độ dài.** Diff: trung vị 917 ký tự, p90 1.472, tối đa 10.363. **Message rất ngắn:** trung vị 6 từ, p90 11, tối đa 24 (gần như tiêu đề commit, không phải câu tóm tắt dài).

**Độ dài token (đã đo trên Kaggle bằng tokenizer Llama 2 thật).** Audit dùng 4.611 mẫu train và 518 mẫu test từ dataset gốc. `diff_only` là diff thô; `chsum_prompt` là diff đặt trong template CHSUM tạo commit message. Đây **không phải** template CHCL dài dùng cho phân loại.

| Split         | Dạng input   | Vượt 512 token | Trung bình | Trung vị |   P90 | Tối đa |
| :------------ | :----------- | -------------: | ---------: | -------: | ----: | -----: |
| Train (4.611) | Diff thô     |    322 (6,98%) |      341,5 |      316 |   481 |  2.808 |
| Train (4.611) | Prompt CHSUM |    392 (8,50%) |      359,5 |      334 |   499 |  2.826 |
| Test (518)    | Diff thô     |     34 (6,56%) |      334,8 |      310 | 475,3 |  1.102 |
| Test (518)    | Prompt CHSUM |     40 (7,72%) |      352,8 |      328 | 493,3 |  1.120 |

**Nhận xét.** Trong audit này, template CHSUM làm độ dài trung bình tăng 18 token; tỷ lệ vượt ngưỡng tăng 1,52 điểm phần trăm ở train và 1,16 điểm ở test. Phần lớn mẫu vẫn nằm trong 512 token, nhưng có đuôi dài tới 2.826 token nên cần truncation nhất quán ở train và inference. Số liệu này không xác nhận giả thuyết template CHCL chiếm khoảng 400 token; đó là template khác và cần đo riêng nếu dùng cho task phân loại.

Kết quả đã lưu trong [prompt_length_summary.csv](improve_step1_results/prompt_length_summary.csv), [prompt_token_lengths.csv](improve_step1_results/prompt_token_lengths.csv) và [tokenizer_audit_metadata.json](improve_step1_results/tokenizer_audit_metadata.json). Notebook ghi nhận lần chạy là [tokenizer123.ipynb](tokenizer123.ipynb). Metadata hiện chưa có `model_revision` (đang `null`), vì vậy bước đếm đã hoàn tất nhưng yêu cầu khóa commit hash của tokenizer **chưa** hoàn tất; không nên xem số liệu này là đã tái lập chính xác theo một revision cố định.

**RQ0 sơ bộ (định danh dạng code trong message so với diff đầy đủ):**

| Chỉ số                                                | Giá trị                        |
| :---------------------------------------------------- | :----------------------------- |
| Message có định danh dạng code (camelCase/snake_case) | 35,1% (1.800 mẫu)              |
| Trong đó có ≥1 định danh **không** có trong diff      | 9,9% (~3,5% toàn bộ mẫu)       |
| Định danh không được hỗ trợ / tổng định danh          | 8,3%                           |
| Theo độ dài diff (tứ phân vị)                         | 9,3–10,4% (gần như không đổi)  |
| Phủ từ nội dung của message trong diff                | trung vị 33%; 64% mẫu dưới 50% |

Đọc nhanh các ví dụ "không được hỗ trợ": phần lớn là tên khái niệm/lớp hợp lệ (`JavaDoc`, `JUnit`, `ClassCastException`), **không phải bịa**. Chưa gán nhãn thủ công nên chưa kết luận chắc.

### 1.3 Điều chỉnh nhận định so với bản README trước

1. **Truncation:** không phải "gần như mọi diff bị cắt". Template cố định mới là nguồn chính của việc vượt 512 token.
2. **Evidence theo định danh yếu hơn dự tính:** chỉ phủ khoảng 1/3 mẫu và bắt được rất ít lỗi bịa thật. Cần định nghĩa lại evidence và thước đo trung thực.
3. **Message ngắn:** câu đích dài ~6 từ, nên ROUGE/BLEU càng nhiễu và hallucination nếu có sẽ ở mức khẳng định/từ ngữ, không chỉ ở định danh.

---

## 2. Mục tiêu và đề xuất (tóm tắt)

**Bài toán.** Input: một code diff (bỏ template). Output: commit message ngắn mô tả thay đổi. Vấn đề: model nhỏ có thể viết câu trôi chảy nhưng **thêm thông tin diff không hỗ trợ**; ROUGE/BLEU không phát hiện được.

**Mục tiêu.** Kiểm chứng xem một phương pháp huấn luyện nhẹ cho model nhỏ, trong đó loss được điều chỉnh dựa trên **bằng chứng trong diff nhìn thấy** (và tùy chọn: ví dụ âm gần giống), có **giảm thông tin không được hỗ trợ mà vẫn giữ chất lượng** hay không. Kết quả âm hoặc đánh đổi đều báo cáo.

**Đề xuất (giữ dữ liệu, model, quy trình fine-tune HQCM; chỉ đổi loss):**

- **Phân tích tiền đề (RQ0):** đo nội dung không được diff hỗ trợ trong chính tham chiếu và trong output của model baseline.
- **Evidence-weighted CE:** tăng trọng số token có bằng chứng trong diff nhìn thấy (trọng số chuẩn hóa để tổng không đổi). _Định nghĩa evidence sẽ chốt sau khi có kết quả gán nhãn thủ công (mục 3, bước 3)._
- **Span-level unlikelihood (mở rộng, nếu còn thời gian):** phạt span khẳng định không được diff hiện tại hỗ trợ, negative lấy từ diff lân cận (BM25/TF-IDF) lọc theo nhóm thay đổi, chỉ lấy từ tập train.
- **Hàm mục tiêu (dự kiến):** `L = L_evidence-CE + λ · L_span-UL`, λ và α chọn trên validation tách từ train.

| RQ  | Câu hỏi                                                                                             | Giai đoạn |
| :-- | :-------------------------------------------------------------------------------------------------- | :-------- |
| RQ0 | Tham chiếu và output model có bao nhiêu nội dung không được diff hỗ trợ?                            | Đang làm  |
| RQ1 | Evidence-weighted CE có tăng độ trung thực so với CE chuẩn **và so với trọng số ngẫu nhiên** không? | Tiếp theo |
| RQ2 | Hard-negative UL có giảm khẳng định sai tốt hơn negative ngẫu nhiên không?                          | Mở rộng   |
| RQ3 | Kết hợp hai thành phần có giảm bịa mà giữ chất lượng không?                                         | Mở rộng   |

**Cấu hình so sánh:** CE baseline · Random-weighted CE (đối chứng bắt buộc) · Evidence-weighted CE · (mở rộng) CE + UL, Full model, CE trên tham chiếu đã lọc, model thứ hai khác họ.

**Phạm vi đề xuất cho một bài báo:** lõi là _RQ0 + evidence-weighted CE (có đối chứng ngẫu nhiên) + đánh giá độc lập chặt_; UL là phần mở rộng.

---

## 3. Cần làm tiếp (theo thứ tự)

| #   | Việc                                                                                                                                                               | Chi phí    | Điều kiện hoàn thành                                     |
| :-- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- | :------------------------------------------------------- |
| 1   | Đếm token trên Kaggle bằng tokenizer Llama 2; lưu CSV (đã xong, xem kết quả ở mục 1.2); khóa `model_revision` vẫn cần làm                                          | ~15 phút   | Tỉ lệ vượt 512 đã đo; còn thiếu revision cố định         |
| 2   | Gán nhãn thủ công ~100 mẫu: 50 mẫu có định danh "không được hỗ trợ" + 50 mẫu phủ từ thấp. Nhãn: _bịa thật / khái niệm hợp lệ / ngữ cảnh ngoài diff / khác_         | 2–3 giờ    | Có tỉ lệ bịa thật trong tham chiếu                       |
| 3   | **Cổng quyết định A/B (xem dưới)**                                                                                                                                 | –          | Chọn hướng                                               |
| 4   | Tách cặp (diff, message), tạo validation từ train, dựng baseline CE tóm tắt bằng harness chung (một hàm tokenize cho train/infer), model nhỏ 1.5–3B, tập con trước | ~1 ngày    | Output không suy biến, loss không NaN, đo thời gian/VRAM |
| 5   | Đo RQ0 trên **output của baseline** (không chỉ tham chiếu)                                                                                                         | Thấp       | Có tỉ lệ bịa của model baseline                          |
| 6   | Cài random-weighted CE và evidence-weighted CE, chạy ≥1 seed trên tập con                                                                                          | Trung bình | Kết quả sơ bộ RQ1                                        |
| 7   | Đánh giá thủ công độc lập quy mô lớn (≥150–200 mẫu, 2 người, Cohen's κ)                                                                                            | Cao        | Bằng chứng chính cho độ trung thực                       |
| 8   | Mở rộng: ≥3 seed, kiểm định thống kê, model thứ hai, UL                                                                                                            | Cao        | Bản thảo                                                 |

**Cell Kaggle cho bước 1:** [KAGGLE_STEP1.md](KAGGLE_STEP1.md). Cell dùng dataset HQCM gốc trong Kaggle Input, đếm token bằng tokenizer Llama 2 cho diff có/không template và lưu CSV cùng metadata; không cần upload thư mục reproduction 80/20 hay adapter.

**Cổng quyết định A/B (sau bước 2):**

- **Hướng A (giữ đề tài tóm tắt/trung thực):** nếu tỉ lệ bịa thật trong tham chiếu hoặc output baseline đủ đáng kể (ngưỡng tạm: khoảng ≥15% mẫu, chốt cùng cô) → tiếp tục bước 4–8.
- **Hướng B (quay lại phân loại, đã có pipeline):** nếu bịa thật rất hiếm → đóng góp "trung thực" yếu. Chuyển sang cải tiến CHCL: audit cắt/template (bỏ template 400 token để dành chỗ cho diff), chấm nhãn bằng log-likelihood ràng buộc (0 invalid, có xác suất, ECE), xử lý nhầm fix↔refactor (loss có trọng số hoặc phân cấp, audit nhãn), đánh giá chặt (≥3 seed/5-fold, bootstrap CI, McNemar), so với baseline encoder.

**Quy tắc cho mọi lần train:** cổng kiểm tra sau lượt chạy ngắn (output không rỗng/lặp, loss không NaN, mọi lớp được dự đoán ở task phân loại); ghi dtype, lượng tử hóa, seed, revision model vào metadata; baseline so sánh là Run 1 hoặc run đã qua cổng, không dùng Run 2.

## 4. Đánh giá và thống kê

**Tách metric huấn luyện khỏi metric đánh giá** để tránh vòng tròn (loss thưởng định danh có trong diff thì metric chính không được chỉ đo đúng điều đó).

- **Chất lượng:** ROUGE, BLEU, METEOR/BERTScore, và metric HQCM dùng (xác nhận). Lưu ý đích rất ngắn (~6 từ) nên các điểm số này nhiễu.
- **Trung thực tự động:** tỷ lệ định danh/khẳng định không có trong diff nhìn thấy (chỉ báo phụ).
- **Trung thực độc lập (bằng chứng chính):** gán nhãn thủ công _đúng / thiếu ý / bịa ý_ trên ≥150–200 mẫu, hai người độc lập, báo cáo κ. LLM-as-judge hoặc ReFEree chỉ dùng nếu đã kiểm chứng tương quan với nhãn thủ công.
- **Chi phí:** thời gian huấn luyện, VRAM, thời gian suy luận.
- **Thống kê:** ≥3 seed, bootstrap 95% CI, kiểm định cặp, effect size; validation tách từ train; kiểm tra trùng commit/dự án giữa train và test.

## 5. Rủi ro

| Rủi ro                                                               | Cách xử lý                                                       |
| :------------------------------------------------------------------- | :--------------------------------------------------------------- |
| Tỉ lệ bịa thật thấp, đóng góp yếu                                    | Cổng A/B; chuyển hướng B                                         |
| Evidence theo định danh phủ ít mẫu                                   | Định nghĩa lại ở mức từ nội dung/khẳng định sau bước 2           |
| Metric trung thực trùng tín hiệu huấn luyện                          | Đánh giá thủ công độc lập làm bằng chứng chính                   |
| Huấn luyện hỏng ngầm (như `feat`=0)                                  | Cổng kiểm tra sớm, cùng một hàm tokenize                         |
| Đích ngắn làm ROUGE/BLEU nhiễu                                       | Bổ sung metric khác, đánh giá thủ công                           |
| Quy tắc span âm phạt oan; trọng số quá mạnh làm câu cụt              | Quy tắc chặt, đo độ chính xác gắn span, chọn α/λ trên validation |
| Rò rỉ train–test                                                     | Negative chỉ từ train; kiểm tra trùng                            |
| Quota GPU không đủ                                                   | Model nhỏ, tập con, ít epoch (run 7B tốn ~6,7 giờ)               |
| Nguồn của bộ 4.103/1.026 (80/20) so với 4.611/518 chưa được xác nhận | Ghi rõ nguồn từng split trong báo cáo                            |

## 6. Mục tiêu công bố

Sau khi hoàn thành hướng A hoặc B với đánh giá chặt, cân nhắc bản thảo cho tạp chí Scopus phù hợp. Phân hạng (kể cả Q3) cần kiểm tra trên Scopus/SCImago vào thời điểm nộp; không có gì đảm bảo trước kết quả phản biện. Trước khi viết cần rà soát tài liệu có hệ thống để xác nhận tính mới.

## 7. Điều muốn xin ý kiến cô

1. Hướng "đổi loss dựa trên bằng chứng" có đủ chiều sâu khoa học không, khi dữ liệu cho thấy tham chiếu rất ngắn và ít bịa định danh?
2. Ngưỡng nào của tỉ lệ bịa thì cô thấy đáng để theo hướng A?
3. Cách đo trung thực (tự động + thủ công) đã hợp lý chưa? Cô có đề xuất công cụ đánh giá độc lập nào không?
4. Mức "chất lượng không giảm đáng kể" chấp nhận được là bao nhiêu?
5. Phạm vi bản thảo: lõi evidence-weighted CE + RQ0, UL là mở rộng, hay hướng B?

## 8. Câu chốt khi báo cáo

> "Em đã chạy được pipeline HQCM ở task phân loại (Macro-F1 0,841 trên split chính thức) và phát hiện/sửa lỗi cắt đầu vào làm mất nhãn. Em đã audit dữ liệu: commit message rất ngắn và bịa định danh hiếm, nên em đang đo chính xác mức bịa thật trước khi chọn hướng. Nếu đủ đáng kể, em sẽ thử huấn luyện nhẹ dựa trên bằng chứng từ diff ở task tạo message; nếu không, em chuyển sang cải tiến độ tin cậy của phân loại. Em xin ý kiến cô về ngưỡng quyết định."

## Tài liệu cần đọc trước

1. Liu, Lin & Thongtanunam (2025) — _Hallucinations in Code Change to Natural Language Generation._
2. Song et al. (2023) — _Model Intrinsic Features of Fine-tuning based Text Summarization Models for Factual Consistency._
3. Nan et al. (2021) — _Improving Factual Consistency of Abstractive Summarization via Question Answering._
4. Bae et al. (2026) — _ReFEree._
5. HQCM — _Understanding Code Changes Practically with Small-Scale Language Models_ (ASE 2024) và repo `codefuse-ai/codefuse-hqcm`.

> Ghi chú trung thực: các tài liệu trên mới được xác nhận ở mức tên, abstract và metadata, **chưa đọc toàn văn**. Hãy ghi lại những gì đã đọc thật.
