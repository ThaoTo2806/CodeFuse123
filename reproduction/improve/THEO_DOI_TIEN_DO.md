# Theo dõi tiến độ reproduction và cải tiến HQCM

Tài liệu này là nơi ghi trạng thái và kết quả chính. Mỗi kết quả mới cần có ngày chạy, cấu hình, split/seed, metric, kết luận giới hạn và đường dẫn artifact. Các cell chi tiết vẫn nằm ở tài liệu riêng được liên kết bên dưới.

## Đọc nhanh

Hướng hiện hành là CHCL. C2/C5/C4, final test CE `both`, C6 encoder và paired validation analysis đã chạy. Còn đánh giá ngoài HQCM/project, related-work/novelty review và hoàn thiện báo cáo.

## Đã làm

1. **Tái hiện pipeline HQCM:** lần đầu train lỗi vì cắt mất nhãn ở cuối mẫu. Đã sửa preprocessing để giữ nhãn và EOS, cắt phần đầu prompt khi cần. Run 1 trên test chính thức 518 mẫu đạt Accuracy 0,7761 và Macro-F1 0,8410 (8 nhãn); dùng làm baseline reproduction. Run 2 trên split 80/20 có lỗi nghiêm trọng ở lớp `feat`, không dùng làm baseline.
2. **Kiểm tra dữ liệu:** 4.611 mẫu train, 518 test chính thức; message tiếng Anh ngắn (trung vị 6 từ). Khoảng 7–8% diff vượt 512 token. Bộ 100 mẫu hỗ trợ gán nhãn thủ công đã được tạo, nhưng việc gán nhãn và tổng hợp của con người chưa hoàn tất.
3. **C3, dự đoán trong tập 8 nhãn:** dùng adapter Run 1 và chấm log-likelihood thay vì sinh tự do. Trên test: Accuracy 0,7799, Macro-F1 0,8384, invalid 0/518, ECE thô 0,1948. Chất lượng gần Run 1; invalid bằng 0 là do cách chấm chỉ cho phép 8 nhãn. ECE chưa hiệu chỉnh.
4. **C1, prompt dài so với compact:** cùng Qwen2.5-Coder-1.5B-Instruct, seed 0, split validation 4.149/462, hai epoch. Kết quả compact tốt hơn trong lần chạy này:
   | Seed | Adapter CE `both` | Accuracy | Macro-F1 (8 nhãn) | Macro-P/R | Weighted-F1 | ECE thô |
   | --: | :-- | --: | --: | --: | --: | --: |
   | 0 | `c2_both_seed0` | 0,8127 | 0,8583 | 0,8819 / 0,8459 | 0,8130 | 0,0973 |
   | 1 | `c5_both_seed1` | 0,8031 | 0,8629 | 0,8688 / 0,8601 | 0,8039 | 0,0886 |
   | 2 | `c5_both_seed2` | 0,7857 | 0,8572 | 0,8693 / 0,8494 | 0,7868 | 0,0921 |
   | ECE thô | 0,4032 | 0,0820 |
   | Prompt trung bình trước cắt | 603,7 token | 304,7 token |
   | Thời gian train | 5.374 giây | 4.575 giây |
   Đã đánh giá test chính thức 518 mẫu với Qwen2.5-Coder-1.5B-Instruct, standard CE, input `both`, revision `2e1fd397...b0f1098a`. Dataset hashes khớp: train `a6430d65...f44c543`, test `c3c804de...f43b78d2`. Seed 0/1/2 hiện đều dùng adapter `both` tương ứng.
   Đây là pilot một seed, không khẳng định template dài là nguyên nhân duy nhất. Xem [kế hoạch và nhận xét C1](README_HUONG_B_CHCL.md), [metrics original](hqcm_c1_prompt_ablation/hqcm_c1_prompt_ablation/c1_original_seed0/metrics.json) và [metrics compact](hqcm_c1_prompt_compact_ablation/hqcm_c1_prompt_compact_ablation/c1_compact_seed0/metrics.json).
   | Seed | Adapter CE `both` | Accuracy | Macro-F1 (8 nhãn) | Macro-P/R | Weighted-F1 | ECE thô |
   | --: | :-- | --: | --: | :-- | --: | --: |
   | 0 | `c2_both_seed0` | 0,8127 | 0,8583 | 0,8819 / 0,8459 | 0,8130 | 0,0973 |
   | 1 | `c5_both_seed1` | 0,8031 | 0,8629 | 0,8688 / 0,8601 | 0,8039 | 0,0886 |
   | 2 | `c5_both_seed2` | 0,7857 | 0,8572 | 0,8693 / 0,8494 | 0,7868 | 0,0921 |
   | Đầu vào | Accuracy | Macro-F1 (8 nhãn) | F1 `fix` | F1 `feat` | F1 `refactor` | ECE thô | Prompt trung bình | Train | VRAM đỉnh |
   | :------------- | ---------: | ----------------: | ---------: | ---------: | ------------: | ---------: | ----------------: | ---------: | --------: |
   | Summary-only | 0,6753 | 0,7390 | 0,6186 | 0,6730 | 0,6985 | 0,0967 | 39,9 token | 4.322 giây | 2,14 GB |
   Ở seed 0/1, F1 `fix` lần lượt 0,7473/0,7579 và F1 `refactor` 0,7770/0,7517. Các số test này không chứng minh cải thiện hơn baseline paper: backbone và protocol khác Run 1/C3. `invalid_count = 0` do constrained scoring chỉ xét 8 nhãn; ECE là thô, chưa calibration. Do đã xem test metrics, việc chấm lại đúng adapter seed 2 phải được ghi minh bạch là sửa lỗi adapter đã định trước, không phải lựa chọn/tinh chỉnh theo test.
   | Diff + summary | **0,7879** | **0,8314** | **0,7500** | **0,8100** | **0,7399** | **0,0935** | 304,7 token | 4.839 giây | 7,25 GB |

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

## Kết quả C4: weighted CE

Đã hoàn tất ngày 31/10/2026 trên cùng validation split, model revision, đầu vào `both` và train seeds 0/1/2. Baseline là CE chuẩn của C2/C5 cùng seed. Weighted CE dùng trọng số nghịch tần suất `N / (8 × số mẫu lớp)`; vì vậy các lớp ít mẫu `style`, `build`, `cicd` có trọng số khoảng 6,4, trong khi `refactor` khoảng 0,44.

|          Seed |     CE Accuracy | Weighted Accuracy | Δ Accuracy |     CE Macro-F1 | Weighted Macro-F1 | Δ Macro-F1 | CE F1 `fix` → weighted | CE F1 `refactor` → weighted |
| ------------: | --------------: | ----------------: | ---------: | --------------: | ----------------: | :--------- | :--------------------- | :-------------------------- |
|             0 |          0,7879 |            0,7900 |    +0,0022 |          0,8314 |            0,8519 | +0,0205    | 0,7500 → 0,7529        | 0,7399 → 0,7557             |
|             1 |          0,7965 |            0,7987 |    +0,0022 |          0,8503 |            0,8589 | +0,0086    | 0,7480 → 0,7490        | 0,7786 → 0,7615             |
|             2 |          0,8182 |            0,7900 |    −0,0281 |          0,8607 |            0,8456 | −0,0152    | 0,7717 → 0,7433        | 0,8045 → 0,7481             |
| Mean ± SD mẫu | 0,8009 ± 0,0156 |   0,7929 ± 0,0050 |    −0,0079 | 0,8475 ± 0,0149 |   0,8521 ± 0,0067 | +0,0046    | 0,7566 → 0,7484 (mean) | 0,7743 → 0,7551 (mean)      |

**Nhận xét:** Weighted CE tăng Macro-F1 ở seed 0 và 1 nhưng giảm ở seed 2; Accuracy giảm rõ ở seed 2. Trung bình ba seed, Macro-F1 chỉ tăng khoảng **0,46 điểm phần trăm**, trong khi Accuracy giảm khoảng **0,79 điểm phần trăm**. Quan trọng với mục tiêu C4, F1 trung bình của `fix` giảm khoảng **0,81 điểm** và `refactor` giảm khoảng **1,92 điểm** so với CE. Do đó, cách đặt trọng số nghịch tần suất đầy đủ **không đạt mục tiêu cải thiện ổn định `fix`/`refactor` và không được chọn làm cấu hình cuối**.

Kết quả validation vẫn chỉ có ba seed trên một split. Bootstrap CI của mỗi run là theo mẫu validation, không thay thế kiểm định giữa các seed; chưa làm kiểm định paired. ECE thô và invalid bằng 0 có cùng giới hạn như các thí nghiệm trước. Cảnh báo CUDA stream không làm các lượt chạy dừng; ba run đều hoàn tất scoring và lưu artifacts.

Artifacts: [C4 summary CSV](hqcm_c4_weighted_ce/hqcm_c4_weighted_ce/all_experiments_summary.csv), [seed 0 metrics](hqcm_c4_weighted_ce/hqcm_c4_weighted_ce/c4_weighted_ce_seed0/metrics.json), [seed 1 metrics](hqcm_c4_weighted_ce/hqcm_c4_weighted_ce/c4_weighted_ce_seed1/metrics.json), [seed 2 metrics](hqcm_c4_weighted_ce/hqcm_c4_weighted_ce/c4_weighted_ce_seed2/metrics.json). Predictions và manifest được lưu cùng từng run. Cell/config: [KAGGLE_C4_WEIGHTED_CE.md](KAGGLE_C4_WEIGHTED_CE.md).

## Kết quả C6: baseline encoder

Đã hoàn tất ngày 31/10/2026 trên cùng split stratified seed 42 (train/validation 4.149/462), ba train seed, dataset hash giống C2/C5. CodeBERT revision `3b0952fe...75e7eb39`; GraphCodeBERT revision `2b0488a7...b0da269d`. Mỗi encoder chạy `diff_only` và `both`; chỉ dùng validation.

| Model         | Input     | Accuracy mean ± SD mẫu | Macro-F1 mean ± SD mẫu | F1 `fix` mean | F1 `refactor` mean |
| :------------ | :-------- | ---------------------: | ---------------------: | ------------: | -----------------: |
| CodeBERT      | diff-only |        0,6905 ± 0,0000 |        0,6508 ± 0,0293 |        0,6268 |             0,6347 |
| CodeBERT      | both      |        0,6905 ± 0,0187 |        0,6587 ± 0,0573 |        0,6242 |             0,6421 |
| GraphCodeBERT | diff-only |        0,7172 ± 0,0054 |        0,7342 ± 0,0113 |        0,6310 |             0,6541 |
| GraphCodeBERT | both      |        0,7222 ± 0,0090 |        0,7443 ± 0,0203 |        0,6376 |             0,6608 |

**Nhận xét:** GraphCodeBERT là encoder tốt hơn trong các cấu hình này, đạt Macro-F1 validation trung bình 0,7443 với `both`; CodeBERT đạt 0,6587. Trong từng encoder, thêm summary chỉ tăng Macro-F1 trung bình nhẹ: khoảng **+0,0080** với CodeBERT và **+0,0101** với GraphCodeBERT. Accuracy CodeBERT không đổi theo mean; GraphCodeBERT tăng khoảng +0,0051. Seed-level effects không đồng nhất, nên không kết luận message luôn có ích ở mọi kiến trúc.

Qwen C2/C5 với `both` đạt Macro-F1 validation 0,8475 ± 0,0149 và Accuracy 0,8009 ± 0,0156, cao hơn các encoder trong protocol này. Đây là baseline tham chiếu thực nghiệm có ích, nhưng **không phải so sánh hoàn toàn cùng serialization**: encoder nhận cặp chuỗi diff/summary, còn Qwen nhận instruction prompt; cách train và model size cũng khác. Cell dùng cùng dữ liệu/split/nhãn và epochs/seed nhưng không loại bỏ các khác biệt kiến trúc này. Các lớp hiếm chỉ có 9 mẫu validation; CodeBERT seed 0 không dự đoán `style`, nên mean per-class cần diễn giải thận trọng.

C6 trả lời một phần RQ4 về baseline encoder trên cùng HQCM validation, nhưng **không trả lời cross-project generalization**. Nó cũng làm RQ1 tinh tế hơn: Qwen hưởng lợi rõ từ `both` trên validation, trong khi hai encoder cho cải thiện nhỏ/không ổn định. Không chọn thêm cấu hình bằng test.

Artifacts: [C6 summary CSV](hqcm_c6_encoder_baselines/all_experiments_summary.csv), cùng metrics/predictions/manifest từng run trong các thư mục `c6_*`. Cell: [KAGGLE_C6_ENCODER_BASELINES.md](KAGGLE_C6_ENCODER_BASELINES.md).

## Kết quả paired validation: `both` so với `diff_only`

Đã ghép validation predictions theo cùng 462 sample IDs cho từng model/seed. Không train lại và không đọc test. Các số sau là chênh lệch paired `both − diff_only`; SD là độ lệch chuẩn mẫu của ba chênh lệch seed.

| Model              | Δ Accuracy mean ± SD | Δ Macro-F1 mean ± SD | Δ F1 `fix` mean | Δ F1 `refactor` mean | Seed `both` Macro-F1 cao hơn |
| :----------------- | -------------------: | -------------------: | --------------: | -------------------: | :--------------------------: |
| Qwen2.5-Coder-1.5B |     +0,0382 ± 0,0033 |     +0,0548 ± 0,0166 |         +0,0409 |              +0,0611 |             3/3              |
| CodeBERT           |      0,0000 ± 0,0187 |     +0,0080 ± 0,0285 |         −0,0026 |              +0,0074 |             2/3              |
| GraphCodeBERT      |     +0,0051 ± 0,0144 |     +0,0101 ± 0,0201 |         +0,0066 |              +0,0068 |             2/3              |

**Nhận xét:** `both` cải thiện nhất quán trên Qwen: cả ba seed đều tăng Macro-F1, với mức tăng trung bình 5,48 điểm phần trăm. Paired bootstrap CI của Δ Macro-F1 theo từng seed chứa 0 ở seed 0, nhưng không chứa 0 ở seed 1/2. McNemar exact p lần lượt là 0,0805 / 0,0414 / 0,0183. Đây là các kiểm định riêng từng seed, chưa hiệu chỉnh cho nhiều phép so sánh; không nên diễn giải như một kiểm định tổng thể.

Với CodeBERT và GraphCodeBERT, mức tăng trung bình nhỏ và không ổn định qua seed. CodeBERT tăng Macro-F1 ở 2/3 seed; GraphCodeBERT cũng 2/3. McNemar p-value theo seed, chưa hiệu chỉnh đa kiểm định: CodeBERT 0,4583 / 0,4244 / 0,0414; GraphCodeBERT 0,2005 / 0,3593 / 0,5572. Kết quả gợi ý lợi ích từ summary phụ thuộc backbone; không đủ để tuyên bố message luôn có ích hoặc có significance tổng quát.

**Giới hạn:** paired bootstrap là hậu nghiệm trên một validation split, có điều kiện trên các model đã train; ba seed không phải 1.386 quan sát độc lập. Không có kiểm định tổng hợp qua seed, hiệu chỉnh đa kiểm định hay cross-project validation. Serialization Qwen và encoder cũng khác nhau.

Artifacts: [paired summary CSV](hqcm_paired_validation_analysis/paired_results_summary.csv), [paired per-seed CSV](hqcm_paired_validation_analysis/paired_results_by_seed.csv), confusion matrices theo model/seed trong cùng thư mục. Cell: [KAGGLE_PAIRED_VALIDATION_ANALYSIS.md](KAGGLE_PAIRED_VALIDATION_ANALYSIS.md).

## Việc tiếp theo

1. **Tăng bằng chứng ngoài HQCM:** tìm/chuẩn bị dataset hoặc project độc lập có nhãn tương thích để đánh giá cross-project; raw HQCM hiện không có repository ID để tự tạo split theo project.
2. **Rà soát novelty và hoàn thiện manuscript:** đối chiếu related work về commit classification, input modalities, constrained scoring và code language models; công khai protocol deviation final test và khác biệt serialization/backbone khi so Qwen với encoder. Chưa khẳng định Q3 cho đến khi các bước này hoàn tất.

Cell C2: [hướng dẫn Kaggle](KAGGLE_C2_INPUT_ABLATION.md). Cell C5: [hướng dẫn và cấu hình tái lập](KAGGLE_C5_SEED_CONFIRMATION.md). C4: [weighted CE](KAGGLE_C4_WEIGHTED_CE.md). C6 encoder: [cell](KAGGLE_C6_ENCODER_BASELINES.md). Paired validation analysis đã hoàn tất; final test CE `both` cũng đã hoàn tất. Không chạy thêm test để chọn mô hình.

## Kết quả test cuối: CE `both`, ba seed

Đã đánh giá test chính thức 518 mẫu với Qwen2.5-Coder-1.5B-Instruct, standard CE, input `both`, revision `2e1fd397...b0f1098a`. Dataset hashes khớp: train `a6430d65...f44c543`, test `c3c804de...f43b78d2`. Lượt đầu seed 2 bị trỏ nhầm sang diff-only; sau đó đã chạy lại seed 2 bằng adapter CE `both` đã chọn trước trên validation. Bảng dưới dùng kết quả đúng cấu hình cho cả ba seed.

|          Seed | Adapter CE `both`   |        Accuracy | Macro-F1 (8 nhãn) |                         Macro-P/R |     Weighted-F1 |         ECE thô |
| ------------: | :------------------ | --------------: | ----------------: | --------------------------------: | --------------: | --------------: |
|             0 | `c2_both_seed0`     |          0,8127 |            0,8583 |                   0,8819 / 0,8459 |          0,8130 |          0,0973 |
|             1 | `c5_both_seed1`     |          0,8031 |            0,8629 |                   0,8688 / 0,8601 |          0,8039 |          0,0886 |
|             2 | `c5_both_seed2`     |          0,7857 |            0,8572 |                   0,8693 / 0,8494 |          0,7868 |          0,0921 |
| Mean ± SD mẫu | Ba seed đã xác minh | 0,8005 ± 0,0137 |   0,8595 ± 0,0031 | 0,8733 ± 0,0074 / 0,8518 ± 0,0074 | 0,8012 ± 0,0133 | 0,0927 ± 0,0044 |

Macro-F1 từng seed nằm trong khoảng 0,8572–0,8629; SD giữa ba seed là 0,0031. F1 `fix` theo seed là 0,7473 / 0,7579 / 0,7305; F1 `refactor` là 0,7770 / 0,7517 / 0,7458. ECE trung bình là số thô chưa calibration. `invalid_count = 0` do constrained scoring chỉ xét tám nhãn.

**Giới hạn protocol:** test đã bị chấm một lần bằng adapter diff-only ở seed 2 trước khi lỗi đường dẫn được phát hiện, sau đó mới chấm seed 2 bằng adapter `both` đã được xác định trước. Kết quả báo cáo dùng adapter đúng, nhưng phải công khai lần chạy nhầm này; đây không còn là một lần đánh giá test hoàn toàn mù. Không dùng test để thay đổi cấu hình. So sánh với Run 1/C3 cũng cần nêu khác biệt backbone và protocol; không diễn giải đây là kết luận vượt paper.

Artifacts đã cập nhật: [final test summary CSV](hqcm_final_test_both/hqcm_final_test_both/all_seed_test_summary.csv), [seed 0 metrics](hqcm_final_test_both/hqcm_final_test_both/seed0_metrics.json), [seed 1 metrics](hqcm_final_test_both/hqcm_final_test_both/seed1_metrics.json), [seed 2 metrics đã sửa](hqcm_final_test_both/hqcm_final_test_both/seed2_metrics.json), [aggregate JSON đã sửa](hqcm_final_test_both/hqcm_final_test_both/all_seed_test_metrics.json). Mỗi run có test scores và confusion matrix trong cùng thư mục. Giữ log lần chạy nhầm kèm hồ sơ nghiên cứu để minh bạch.

## Nhật ký kết quả

| Ngày       | Thí nghiệm           | Cấu hình/split                                       | Kết quả chính                                                              | Trạng thái / ghi chú                            |
| :--------- | :------------------- | :--------------------------------------------------- | :------------------------------------------------------------------------- | :---------------------------------------------- |
| 28/10/2026 | Reproduction Run 1   | Llama-2-7B QLoRA; test 518                           | Accuracy 0,7761; Macro-F1 0,8410                                           | Baseline; 1 output invalid                      |
| 28/10/2026 | C3                   | Adapter Run 1; test 518; constrained scoring         | Accuracy 0,7799; Macro-F1 0,8384; ECE 0,1948                               | Hoàn tất; ECE thô                               |
| 29/10/2026 | C1 original          | Qwen2.5-Coder-1.5B; seed 0; split 4.149/462          | Accuracy 0,3052; Macro-F1 0,2611                                           | Pilot; revision chưa khóa                       |
| 29/10/2026 | C1 compact           | Cùng cấu hình/split với original                     | Accuracy 0,7900; Macro-F1 0,8560                                           | Pilot; revision chưa khóa                       |
| 30/10/2026 | C2 summary-only      | Qwen2.5-Coder-1.5B; rev `2e1fd397`; seed 0/42        | Accuracy 0,6753; Macro-F1 0,7390; ECE 0,0967                               | Validation; 1 seed                              |
| 30/10/2026 | C2 diff-only         | Cùng revision/split/seed                             | Accuracy 0,7532; Macro-F1 0,7957; ECE 0,1195                               | Validation; 1 seed                              |
| 30/10/2026 | C2 both              | Cùng revision/split/seed                             | Accuracy 0,7879; Macro-F1 0,8314; ECE 0,0935                               | Validation; 1 seed                              |
| 30/10/2026 | C5 diff-only         | Seeds 0/1/2; cùng split và revision                  | Mean Acc 0,7626 ± 0,0127; Macro-F1 0,7927 ± 0,0055                         | Validation; 3 seeds                             |
| 30/10/2026 | C5 both              | Seeds 0/1/2; cùng split và revision                  | Mean Acc 0,8009 ± 0,0156; Macro-F1 0,8475 ± 0,0149                         | Tốt hơn diff-only ở cả ba seed                  |
| 31/10/2026 | C4 weighted CE       | Đầu vào `both`; seeds 0/1/2; cùng split/revision     | Mean Acc 0,7929 ± 0,0050; Macro-F1 0,8521 ± 0,0067                         | Không chọn: F1 `fix`/`refactor` giảm trung bình |
| 31/10/2026 | Test CE `both`       | Seeds 0/1/2; seed 2 đã chấm lại adapter đúng         | Mean Acc 0,8005 ± 0,0137; Macro-F1 0,8595 ± 0,0031                         | Ba seed hợp lệ; ghi protocol deviation lần đầu  |
| 31/10/2026 | C6 encoder baselines | CodeBERT/GraphCodeBERT; 2 input × 3 seed; validation | Macro-F1: CB diff 0,6508; CB both 0,6587; GCB diff 0,7342; GCB both 0,7443 | Hoàn tất; input effect nhỏ/không đồng nhất      |
| 31/10/2026 | Paired validation    | C2/C5/C6 predictions; 462 IDs ghép cặp; 3 seed/model | Δ Macro-F1: Qwen +0,0548; CodeBERT +0,0080; GraphCodeBERT +0,0101          | Qwen tăng 3/3; encoder nhỏ/không ổn định        |
| Tiếp theo  | Cross-project        | Dataset/project độc lập có type labels tương thích   | Chưa có                                                                    | Khoảng trống chính trước khi nộp                |

## Quy tắc ghi kết quả mới

Ghi thêm một dòng nhật ký sau mỗi lần chạy: ngày; mã/variant; model và revision nếu có; split và seed; Accuracy, Macro-F1 và F1 từng lớp; ECE và ghi chú calibration; thời gian; đường dẫn artifact; vấn đề hoặc giới hạn. Tách rõ kết quả validation với test. Chỉ mở test chính thức cho cấu hình đã chọn, không dùng test để lựa chọn.

## Tệp tham khảo

- [Báo cáo reproduction](../doc/README_step1.md)
- [Chi tiết hướng cải tiến CHCL](README_HUONG_B_CHCL.md)
- [Cell C1 prompt ablation](KAGGLE_C1_PROMPT_ABLATION.md)
- [Cell C3 constrained scoring](README_HUONG_B_CHCL.md#4-cell-kaggle-ngay-1-c3-tren-adapter-run-1)
- [Audit độ dài token](KAGGLE_STEP1.md)
