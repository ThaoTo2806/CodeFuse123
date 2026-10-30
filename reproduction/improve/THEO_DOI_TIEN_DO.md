# Theo dõi tiến độ reproduction và cải tiến HQCM

Tài liệu này là nơi ghi trạng thái và kết quả chính. Mỗi kết quả mới cần có ngày chạy, cấu hình, split/seed, metric, kết luận giới hạn và đường dẫn artifact. Các cell chi tiết vẫn nằm ở tài liệu riêng được liên kết bên dưới.

## Đọc nhanh

Bài toán đang làm là **phân loại loại commit** thành 8 nhãn (`style`, `docs`, `test`, `build`, `cicd`, `fix`, `feat`, `refactor`). Dữ liệu HQCM có sẵn diff (`change`), commit message (`summaries.en`) và nhãn (`type`). Ta đang kiểm tra xem bỏ template dài hay bỏ bớt một nguồn thông tin có giúp phân loại tốt hơn không.

## Đã làm

1. **Tái hiện pipeline HQCM:** lần đầu train lỗi vì cắt mất nhãn ở cuối mẫu. Đã sửa preprocessing để giữ nhãn và EOS, cắt phần đầu prompt khi cần. Run 1 trên test chính thức 518 mẫu đạt Accuracy 0,7761 và Macro-F1 0,8410 (8 nhãn); dùng làm baseline reproduction. Run 2 trên split 80/20 có lỗi nghiêm trọng ở lớp `feat`, không dùng làm baseline.
2. **Kiểm tra dữ liệu:** 4.611 mẫu train, 518 test chính thức; message tiếng Anh ngắn (trung vị 6 từ). Khoảng 7–8% diff vượt 512 token. Bộ 100 mẫu hỗ trợ gán nhãn thủ công đã được tạo, nhưng việc gán nhãn và tổng hợp của con người chưa hoàn tất.
3. **C3, dự đoán trong tập 8 nhãn:** dùng adapter Run 1 và chấm log-likelihood thay vì sinh tự do. Trên test: Accuracy 0,7799, Macro-F1 0,8384, invalid 0/518, ECE thô 0,1948. Chất lượng gần Run 1; invalid bằng 0 là do cách chấm chỉ cho phép 8 nhãn. ECE chưa hiệu chỉnh.
4. **C1, prompt dài so với compact:** cùng Qwen2.5-Coder-1.5B-Instruct, seed 0, split validation 4.149/462, hai epoch. Kết quả compact tốt hơn trong lần chạy này:

| Chỉ số validation           |    Original |     Compact |
| :-------------------------- | ----------: | ----------: |
| Accuracy                    |      0,3052 |      0,7900 |
| Macro-F1 (8 nhãn)           |      0,2611 |      0,8560 |
| ECE thô                     |      0,4032 |      0,0820 |
| Prompt trung bình trước cắt | 603,7 token | 304,7 token |
| Thời gian train             |  5.374 giây |  4.575 giây |

Đây là pilot một seed, không khẳng định template dài là nguyên nhân duy nhất. Xem [kế hoạch và nhận xét C1](README_HUONG_B_CHCL.md), [metrics original](hqcm_c1_prompt_ablation/hqcm_c1_prompt_ablation/c1_original_seed0/metrics.json) và [metrics compact](hqcm_c1_prompt_compact_ablation/hqcm_c1_prompt_compact_ablation/c1_compact_seed0/metrics.json).

## Kết quả mới nhất: C2 input ablation

Đã hoàn tất ngày 30/10/2026 trên validation: Qwen2.5-Coder-1.5B-Instruct, revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, seed 0, split seed 42, train/validation 4.149/462, cùng hash dữ liệu train. Ba nhánh dùng chung cấu hình QLoRA 4-bit, hai epoch và cách chấm điểm tập đóng tám nhãn.

| Đầu vào        |   Accuracy | Macro-F1 (8 nhãn) |   F1 `fix` |  F1 `feat` | F1 `refactor` |    ECE thô | Prompt trung bình |      Train | VRAM đỉnh |
| :------------- | ---------: | ----------------: | ---------: | ---------: | ------------: | ---------: | ----------------: | ---------: | --------: |
| Summary-only   |     0,6753 |            0,7390 |     0,6186 |     0,6730 |        0,6985 |     0,0967 |        39,9 token | 4.322 giây |   2,14 GB |
| Diff-only      |     0,7532 |            0,7957 |     0,7013 |     0,7596 |        0,7076 |     0,1195 |       292,8 token | 4.764 giây |   6,00 GB |
| Diff + summary | **0,7879** |        **0,8314** | **0,7500** | **0,8100** |    **0,7399** | **0,0935** |       304,7 token | 4.839 giây |   7,25 GB |

**Nhận xét:** Trong lần chạy này, dùng cả diff và message đứng đầu; so với chỉ diff, tăng 0,0346 Accuracy và 0,0357 Macro-F1 (khoảng 3,5 điểm phần trăm). Diff-only tốt hơn summary-only 0,0567 Macro-F1, nên diff là nguồn quan trọng hơn trong ba cấu hình đã thử. Dùng cả hai cũng cho F1 cao nhất trong ba lớp lớn `fix`, `feat`, `refactor`. Lợi ích đi kèm thời gian train và VRAM cao hơn diff-only.

Đây vẫn là **một seed trên một validation split**, chưa có CI; chưa đủ để kết luận message luôn cải thiện kết quả. Kết luận chỉ áp dụng cho prompt, max length và cách train/chấm điểm đã dùng. ECE là thô, chưa calibration; invalid bằng 0 do decoder chỉ chọn tám nhãn. Cảnh báo CUDA stream trong log không làm lượt chạy dừng; cả ba nhánh hoàn tất chấm validation và lưu metrics. C1 compact trước đó có `model_revision = null` và metric khác; không gộp nó với bảng C2 đã khóa revision.

Artifact C2: [bảng tổng hợp](hqcm_c2_input_ablation/hqcm_c2_input_ablation/all_experiments_summary.csv), [summary-only metrics](hqcm_c2_input_ablation/hqcm_c2_input_ablation/c2_summary_only_seed0/metrics.json), [diff-only metrics](hqcm_c2_input_ablation/hqcm_c2_input_ablation/c2_diff_only_seed0/metrics.json), [both metrics](hqcm_c2_input_ablation/hqcm_c2_input_ablation/c2_both_seed0/metrics.json). Mỗi run còn có predictions, manifest và adapter.

## Kết quả C5: xác nhận qua ba seed

Đã hoàn tất ngày 30/10/2026. C5 chạy `diff_only` và `both` với train seed 1/2; kết hợp seed 0 từ C2 để có ba seed trên cùng validation split 462 mẫu. Dataset SHA-256 và model revision được giữ nguyên giữa các run.

| Đầu vào        | Accuracy theo seed 0 / 1 / 2 | Macro-F1 theo seed 0 / 1 / 2 | Trung bình ± SD mẫu Accuracy | Trung bình ± SD mẫu Macro-F1 |
| :------------- | :--------------------------- | :--------------------------- | ---------------------------: | ---------------------------: |
| Diff-only      | 0,7532 / 0,7576 / 0,7771     | 0,7957 / 0,7863 / 0,7960     |              0,7626 ± 0,0127 |              0,7927 ± 0,0055 |
| Diff + summary | 0,7879 / 0,7965 / 0,8182     | 0,8314 / 0,8503 / 0,8607     |              0,8009 ± 0,0156 |              0,8475 ± 0,0149 |

Ở từng seed, `both` đều cao hơn `diff_only`. Chênh lệch trung bình theo cặp seed là **+0,0382 Accuracy** (3,82 điểm phần trăm) và **+0,0548 Macro-F1** (5,48 điểm phần trăm). F1 `fix` và `refactor` cũng cao hơn với `both` ở cả ba seed; đây là tín hiệu phù hợp với mục tiêu giảm nhầm lẫn giữa các lớp chính.

**Diễn giải:** C5 củng cố lựa chọn `both` làm đầu vào ứng viên cho bước tiếp theo. Tuy nhiên, ba seed vẫn là số lượng nhỏ và chỉ dùng một validation split; SD giữa seed không phải khoảng tin cậy, còn bootstrap CI trong từng metrics chỉ mô tả bất định theo mẫu validation của run đó. Chưa có kiểm định paired trên từng dự đoán, chưa đánh giá test và chưa thể kết luận có ý nghĩa thống kê/khái quát hóa. ECE tiếp tục là số thô chưa hiệu chỉnh. Cảnh báo CUDA stream xuất hiện trong log nhưng các run hoàn tất scoring và lưu artifact.

Artifacts: [C5 summary CSV](hqcm_c5_seed_confirmation/hqcm_c5_seed_confirmation/all_experiments_summary.csv), [diff-only seed 1](hqcm_c5_seed_confirmation/hqcm_c5_seed_confirmation/c5_diff_only_seed1/metrics.json), [diff-only seed 2](hqcm_c5_seed_confirmation/hqcm_c5_seed_confirmation/c5_diff_only_seed2/metrics.json), [both seed 1](hqcm_c5_seed_confirmation/hqcm_c5_seed_confirmation/c5_both_seed1/metrics.json), [both seed 2](hqcm_c5_seed_confirmation/hqcm_c5_seed_confirmation/c5_both_seed2/metrics.json). Seed 0 nằm trong [artifact C2](hqcm_c2_input_ablation/hqcm_c2_input_ablation/all_experiments_summary.csv). Predictions, split indices và adapter được lưu cùng từng run.

## Việc tiếp theo

1. **C4 pilot trên đầu vào `both`:** so sánh CE chuẩn với một phương pháp xử lý mất cân bằng (ưu tiên weighted CE theo tần suất lớp train) trên cùng split, model revision và seed. Đánh giá Macro-F1 cùng F1 `fix`/`refactor`; không đổi prompt hay scoring cùng lúc để biết thay đổi loss có tác dụng gì. Chỉ nhân thêm seed nếu run qua cổng chất lượng.
2. **Sau khi khóa ứng viên:** chạy đánh giá test chính thức một lần, lưu predictions và so sánh paired với baseline reproduction/C3. Không chọn cấu hình bằng test.

Cell C2: [hướng dẫn Kaggle](KAGGLE_C2_INPUT_ABLATION.md). Cell C5 đã hoàn tất: [hướng dẫn và cấu hình tái lập](KAGGLE_C5_SEED_CONFIRMATION.md). Bước chạy tiếp theo là [cell C4 weighted CE](KAGGLE_C4_WEIGHTED_CE.md): một cell lặp seed 0/1/2; baseline CE tương ứng đã có từ C2/C5. Sau khi chạy, gửi `all_experiments_summary.csv` và metrics/predictions để ghi kết quả vào nhật ký.

## Nhật ký kết quả

| Ngày       | Thí nghiệm         | Cấu hình/split                                | Kết quả chính                                      | Trạng thái / ghi chú              |
| :--------- | :----------------- | :-------------------------------------------- | :------------------------------------------------- | :-------------------------------- |
| 28/10/2026 | Reproduction Run 1 | Llama-2-7B QLoRA; test 518                    | Accuracy 0,7761; Macro-F1 0,8410                   | Baseline; 1 output invalid        |
| 28/10/2026 | C3                 | Adapter Run 1; test 518; constrained scoring  | Accuracy 0,7799; Macro-F1 0,8384; ECE 0,1948       | Hoàn tất; ECE thô                 |
| 29/10/2026 | C1 original        | Qwen2.5-Coder-1.5B; seed 0; split 4.149/462   | Accuracy 0,3052; Macro-F1 0,2611                   | Pilot; revision chưa khóa         |
| 29/10/2026 | C1 compact         | Cùng cấu hình/split với original              | Accuracy 0,7900; Macro-F1 0,8560                   | Pilot; revision chưa khóa         |
| 30/10/2026 | C2 summary-only    | Qwen2.5-Coder-1.5B; rev `2e1fd397`; seed 0/42 | Accuracy 0,6753; Macro-F1 0,7390; ECE 0,0967       | Validation; 1 seed                |
| 30/10/2026 | C2 diff-only       | Cùng revision/split/seed                      | Accuracy 0,7532; Macro-F1 0,7957; ECE 0,1195       | Validation; 1 seed                |
| 30/10/2026 | C2 both            | Cùng revision/split/seed                      | Accuracy 0,7879; Macro-F1 0,8314; ECE 0,0935       | Validation; 1 seed                |
| 30/10/2026 | C5 diff-only       | Seeds 0/1/2; cùng split và revision           | Mean Acc 0,7626 ± 0,0127; Macro-F1 0,7927 ± 0,0055 | Validation; 3 seeds               |
| 30/10/2026 | C5 both            | Seeds 0/1/2; cùng split và revision           | Mean Acc 0,8009 ± 0,0156; Macro-F1 0,8475 ± 0,0149 | Tốt hơn diff-only ở cả ba seed    |
| Tiếp theo  | C4 weighted CE     | Đầu vào `both`; cùng split/revision           | Chờ chạy                                           | Pilot; tập trung `fix`/`refactor` |

## Quy tắc ghi kết quả mới

Ghi thêm một dòng nhật ký sau mỗi lần chạy: ngày; mã/variant; model và revision nếu có; split và seed; Accuracy, Macro-F1 và F1 từng lớp; ECE và ghi chú calibration; thời gian; đường dẫn artifact; vấn đề hoặc giới hạn. Tách rõ kết quả validation với test. Chỉ mở test chính thức cho cấu hình đã chọn, không dùng test để lựa chọn.

## Tệp tham khảo

- [Báo cáo reproduction](../doc/README_step1.md)
- [Chi tiết hướng cải tiến CHCL](README_HUONG_B_CHCL.md)
- [Cell C1 prompt ablation](KAGGLE_C1_PROMPT_ABLATION.md)
- [Cell C3 constrained scoring](README_HUONG_B_CHCL.md#4-cell-kaggle-ngay-1-c3-tren-adapter-run-1)
- [Audit độ dài token](KAGGLE_STEP1.md)
