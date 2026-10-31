# Do Commit Messages Help Commit-Type Classification?

## Quyết định hướng nghiên cứu (31/10/2026)

**Hướng đã chốt: phân loại loại commit (CHCL) bằng small code language models.** Không tiếp tục xem đề tài evidence-aware summarization/hallucination là hướng chính: chưa có gán nhãn thủ công, baseline tóm tắt hay thí nghiệm evidence-weighted loss/negative sampling. Các mục RQ0–RQ3 về summarization ở phần sau được giữ làm ghi chú lịch sử, **không phải kết quả đã trả lời**.

**Câu chuyện nghiên cứu dự kiến:** đo một cách có kiểm soát tác động của nguồn đầu vào (diff, commit message, hoặc cả hai) và constrained likelihood scoring đối với chất lượng/độ hợp lệ của phân loại commit; kiểm tra liệu class-weighted CE có giúp các lớp `fix`/`refactor` hay không. Đây là đóng góp đánh giá thực nghiệm và protocol, chưa phải thuật toán mới.

### Câu hỏi nghiên cứu CHCL

| RQ  | Câu hỏi                                                                      | Bằng chứng hiện có                                                                                                                                                                                     | Trạng thái                                                                           |
| :-- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| RQ1 | Diff, commit message hay cả hai cho kết quả tốt hơn?                         | C2/C5: `both` có Macro-F1 validation trung bình 0,8475 so với 0,7927 của `diff-only` trên 3 seed cùng split. Test cuối chỉ đánh giá `both`; vì vậy kết luận modality trên test chưa có so sánh paired. | Có kết quả validation; cần xác nhận độc lập/generalization                           |
| RQ2 | Constrained scoring có thể loại output không hợp lệ mà giữ chất lượng không? | C3 trên Llama-2 Run 1: Accuracy 0,7799 so với 0,7761; Macro-F1 0,8384 so với 0,8410; invalid 0/518 thay vì 1/518.                                                                                      | Đã trả lời sơ bộ trên một test/split; invalid bằng 0 là đảm bảo của decoder tập đóng |
| RQ3 | Inverse-frequency weighted CE có cải thiện `fix`/`refactor` ổn định không?   | C4 ba seed: Macro-F1 trung bình chỉ tăng nhẹ, Accuracy giảm; F1 `fix` và `refactor` trung bình đều giảm so với CE.                                                                                     | Kết quả âm: không chọn weighted CE                                                   |
| RQ4 | Kết quả có đứng vững trước baseline encoder cùng protocol không?             | Chưa có CodeBERT/GraphCodeBERT trên cùng split.                                                                                                                                                        | Việc chạy tiếp: [cell C6](KAGGLE_C6_ENCODER_BASELINES.md); validation-only           |

### Đóng góp có thể tuyên bố ở thời điểm này

1. Tái hiện CHCL và audit lỗi preprocessing làm mất target; sửa pipeline để giữ nhãn/EOS và báo cáo metric trên đúng 8 lớp.
2. Thực nghiệm nhiều seed trên HQCM cho thấy việc kết hợp diff và commit message là cấu hình ứng viên tốt hơn diff-only trên validation; không tuyên bố quan hệ nhân quả tổng quát.
3. Đánh giá constrained likelihood scoring và báo cáo trade-off chất lượng/invalid; invalid bằng 0 là thuộc tính phương pháp, không phải năng lực sinh tự do.
4. Báo cáo kết quả âm của inverse-frequency weighted CE: không có cải thiện ổn định ở `fix`/`refactor`.

**Đánh giá mức sẵn sàng:** có kết quả nội bộ để viết bản thảo ban đầu, nhưng **chưa đủ để khẳng định sẵn sàng cho Scopus Q3**. Hiện chỉ có một benchmark; chưa có kiểm định paired được chốt, baseline encoder hoặc xác nhận ngoài HQCM; test cuối có protocol deviation vì seed 2 từng bị chấm bằng adapter diff-only trước khi chạy lại đúng adapter. Phải công khai sai lệch này và không tinh chỉnh thêm trên test. Q3 không thể bảo đảm bằng metric; cần rà soát novelty/literature và chọn tạp chí phù hợp.

**Việc cần làm tiếp:** chạy cell C6 để có baseline encoder cùng split; sau đó so sánh validation theo seed và từng lớp. Để tăng sức nặng trước khi nộp, vẫn cần thêm đánh giá ngoài HQCM/project hoặc xác nhận độc lập, hoàn tất related-work/novelty review, và công khai protocol deviation ở final test. Cell C6 không đọc test và lặp 12 cấu hình trong một vòng lặp.

Theo dõi kết quả: [THEO_DOI_TIEN_DO.md](THEO_DOI_TIEN_DO.md). Kế hoạch CHCL chi tiết: [README_HUONG_B_CHCL.md](README_HUONG_B_CHCL.md). Cell final test và ghi chú protocol: [KAGGLE_FINAL_TEST_BOTH.md](KAGGLE_FINAL_TEST_BOTH.md).

**Cell tiếp theo để chạy:** [KAGGLE_C6_ENCODER_BASELINES.md](KAGGLE_C6_ENCODER_BASELINES.md). Cell lặp CodeBERT/GraphCodeBERT × `diff_only`/`both` × seed 0/1/2, tự pin model revisions và lưu metrics/predictions mỗi run.

> **Hướng cũ đã tạm dừng:** Evidence-Aware Fine-Tuning for Faithful Code Change Summarization. Chưa đủ dữ liệu thủ công/baseline để đưa ra đóng góp factuality.

> Ghi chú lưu trữ của hướng summarization: tên _"Functionally Hard Negatives"_ chỉ có ý nghĩa nếu sau này quay lại hướng đó và xác minh negative bằng đánh giá thủ công.

> Trạng thái cũ ngày 29/10/2026 dưới đây đã được thay thế bởi quyết định nghiên cứu ở đầu file và nhật ký [THEO_DOI_TIEN_DO.md](THEO_DOI_TIEN_DO.md). Không dùng các bước/RQ summarization cũ làm kế hoạch hiện hành.

---

## Nội dung lưu trữ: hướng summarization/evidence-aware (đã tạm dừng)

Các mục bên dưới ghi lại giả thuyết và công việc của hướng tóm tắt commit trung thực trước khi nhóm chuyển sang CHCL. Chúng hữu ích làm lịch sử quyết định, nhưng chưa được thực nghiệm đầy đủ và không phải các RQ đã trả lời.

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

### 1.2 Audit dữ liệu HQCM (repo `CodeFuse123/dataset`, 4.611 train / 518 test)

**Cấu trúc thật của dữ liệu.** Mỗi mẫu có 4 trường: `type` (8 nhãn), `subtype` (46 cặp type–subtype, theo phân loại tham khảo từ nghiên cứu code review EMSE20), `change` (diff) và `summaries` (`en` và `zh`). Như vậy **đã có sẵn cặp diff → summary** cho task tạo mô tả, không cần tách từ prompt. Hai file 4.103/1.026 dùng trước đó là bản chuyển sang prompt CHCL và chia lại 80/20; số liệu bên dưới chạy trên dữ liệu gốc.

**Phân bố nhãn (train):** refactor 1.324 · fix 1.191 · feat 1.018 · test 450 · docs 357 · build 91 · cicd 90 · style 90. Test chính thức: refactor 148 · fix 133 · feat 114 · test 50 · docs 40 · build/cicd/style 11 mỗi lớp.

**Summary rất ngắn:** `summaries.en` trung vị 6 từ, p90 11, tối đa 24 (gần như tiêu đề commit). Có 99 summary trùng lặp. Không có diff trùng trong dữ liệu và không có diff test trùng với train.

**Độ dài token (Llama-2 tokenizer, `max_length` 512; từ `prompt_length_summary.csv`):**

| Split | View         | Mẫu   | Vượt 512        | Trung vị | p90 | Tối đa |
| :---- | :----------- | :---- | :-------------- | :------- | :-- | :----- |
| train | diff_only    | 4.611 | 322 (**6,98%**) | 316      | 481 | 2.808  |
| train | chsum_prompt | 4.611 | 392 (8,50%)     | 334      | 499 | 2.826  |
| test  | diff_only    | 518   | 34 (**6,56%**)  | 310      | 475 | 1.102  |
| test  | chsum_prompt | 518   | 40 (7,72%)      | 328      | 493 | 1.120  |

Thêm: chỉ 0,17% diff train vượt 1.024 token; p95 = 546, p99 = 716 token. Wrapper CHSUM chỉ thêm 18 token. **Kết luận: truncation ở 512 chỉ ảnh hưởng khoảng 7–8% mẫu.** Nhận định cũ "gần như mọi mẫu bị cắt" (từ prompt CHCL có template ~400 token) không áp dụng cho task tạo mô tả. Mức này nằm trong khoảng ước lượng 2–10% trước đó.

**RQ0 sơ bộ (định danh dạng code trong `summaries.en` so với `change` đầy đủ, 5.129 mẫu):**

| Chỉ số                                                | Giá trị                       |
| :---------------------------------------------------- | :---------------------------- |
| Summary có định danh dạng code (camelCase/snake_case) | 35,1% (1.800 mẫu)             |
| Trong đó có ≥1 định danh **không** có trong diff      | 9,9% (178 mẫu, ~3,5% toàn bộ) |
| Định danh không được hỗ trợ / tổng định danh          | 8,3%                          |
| Theo độ dài diff (tứ phân vị)                         | 9,3–10,4% (gần như không đổi) |

Đọc nhanh các ví dụ "không được hỗ trợ": phần lớn là tên khái niệm/lớp hợp lệ (`JavaDoc`, `JUnit`, `ClassCastException`), **không phải bịa**. Chưa gán nhãn thủ công nên chưa kết luận chắc. Phủ từ nội dung của summary trong diff: trung vị 33%.

### 1.3 Điều chỉnh nhận định so với bản README trước

1. **Truncation là vấn đề nhỏ** (~7–8% mẫu ở 512 token). Lỗi cắt phải của repo gốc vẫn nghiêm trọng (mất nhãn), nhưng việc nén diff (C2) **không còn là ưu tiên**.
2. **Evidence theo định danh yếu hơn dự tính:** chỉ phủ khoảng 1/3 mẫu và bắt được rất ít lỗi bịa thật. Cần định nghĩa lại evidence và thước đo trung thực.
3. **Summary ngắn:** đích dài ~6 từ nên ROUGE/BLEU nhiễu, và hallucination nếu có sẽ ở mức khẳng định/từ ngữ.
4. **Có `subtype`:** dùng được để lọc negative theo nhóm mịn hơn, và để phân tích nhầm lẫn fix/refactor ở mức subtype.
5. **Còn thiếu:** `model_revision` vẫn `null` và chưa có `tokenizer_fingerprint` (chưa khóa commit hash của model/tokenizer). Chưa rõ summary được tạo bằng cách nào (người viết hay LLM); cần xác nhận từ paper/repo vì ảnh hưởng đến cách diễn giải RQ0.

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

| #   | Việc                                                                                                                                                                                                                         | Chi phí    | Điều kiện hoàn thành                                     |
| :-- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- | :------------------------------------------------------- |
| 1   | ~~Đo token~~ **Xong** (7–8% mẫu vượt 512). Còn lại: khóa `model_revision` (lấy hash bằng `HfApi().model_info(...).sha`) và ghi `tokenizer_fingerprint`                                                                       | ~10 phút   | Metadata không còn `null`                                |
| 2   | Gán nhãn thủ công ~100 mẫu: 50 mẫu có định danh "không được hỗ trợ" + 50 mẫu phủ từ thấp. Nhãn: _bịa thật / khái niệm hợp lệ / ngữ cảnh ngoài diff / khác_                                                                   | 2–3 giờ    | Có tỉ lệ bịa thật trong tham chiếu                       |
| 3   | **Cổng quyết định A/B (xem dưới)**                                                                                                                                                                                           | –          | Chọn hướng                                               |
| 4   | Dùng trực tiếp (`change` → `summaries.en`) từ dataset gốc, tạo validation từ train (stratified theo `type`), dựng baseline CE tóm tắt bằng harness chung (một hàm tokenize cho train/infer), model nhỏ 1.5–3B, tập con trước | ~1 ngày    | Output không suy biến, loss không NaN, đo thời gian/VRAM |
| 5   | Đo RQ0 trên **output của baseline** (không chỉ tham chiếu)                                                                                                                                                                   | Thấp       | Có tỉ lệ bịa của model baseline                          |
| 6   | Cài random-weighted CE và evidence-weighted CE, chạy ≥1 seed trên tập con                                                                                                                                                    | Trung bình | Kết quả sơ bộ RQ1                                        |
| 7   | Đánh giá thủ công độc lập quy mô lớn (≥150–200 mẫu, 2 người, Cohen's κ)                                                                                                                                                      | Cao        | Bằng chứng chính cho độ trung thực                       |
| 8   | Mở rộng: ≥3 seed, kiểm định thống kê, model thứ hai, UL                                                                                                                                                                      | Cao        | Bản thảo                                                 |

**Cell Kaggle cho bước 2:** [KAGGLE_STEP2_MANUAL_LABELING.md](KAGGLE_STEP2_MANUAL_LABELING.md). Cell đã được chạy và CSV mẫu nằm tại `improve_step2_manual_labeling/manual_labels_100.csv`. Cell lấy 50 mẫu có định danh bị flag và 50 mẫu coverage thấp; các cờ tự động chỉ dùng để lấy mẫu, không phải kết luận bịa. **Phần con người gán nhãn 100 mẫu và tổng hợp kết quả vẫn chưa hoàn thành.**

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

> "Em đã chạy được pipeline HQCM ở task phân loại (Macro-F1 0,841 trên split chính thức) và phát hiện/sửa lỗi cắt đầu vào làm mất nhãn. Em đã audit dữ liệu: chỉ 7–8% diff vượt 512 token, summary rất ngắn và bịa định danh hiếm, nên em đang đo chính xác mức bịa thật trước khi chọn hướng. Nếu đủ đáng kể, em sẽ thử huấn luyện nhẹ dựa trên bằng chứng từ diff ở task tạo message; nếu không, em chuyển sang cải tiến độ tin cậy của phân loại. Em xin ý kiến cô về ngưỡng quyết định."

## Tài liệu cần đọc trước

1. Liu, Lin & Thongtanunam (2025) — _Hallucinations in Code Change to Natural Language Generation._
2. Song et al. (2023) — _Model Intrinsic Features of Fine-tuning based Text Summarization Models for Factual Consistency._
3. Nan et al. (2021) — _Improving Factual Consistency of Abstractive Summarization via Question Answering._
4. Bae et al. (2026) — _ReFEree._
5. HQCM — _Understanding Code Changes Practically with Small-Scale Language Models_ (ASE 2024) và repo `codefuse-ai/codefuse-hqcm`.

> Ghi chú trung thực: các tài liệu trên mới được xác nhận ở mức tên, abstract và metadata, **chưa đọc toàn văn**. Hãy ghi lại những gì đã đọc thật.
