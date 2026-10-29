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

## Đang làm tiếp

**C2: đo nguồn thông tin hữu ích.** So sánh `summary-only`, `diff-only`, `diff + summary` trên cùng split, seed, model và cấu hình train. Nhánh `diff + summary` chính là C1 compact seed 0 đã chạy, nên không cần train lại; cell mới chạy hai nhánh còn thiếu. Chạy trên validation, không dùng test để chọn cấu hình.

Cell và hướng dẫn: [KAGGLE_C2_INPUT_ABLATION.md](KAGGLE_C2_INPUT_ABLATION.md). Kết quả C2 sẽ được ghi tại mục dưới đây và lưu metrics/predictions/manifest trong Kaggle Output.

## Nhật ký kết quả

| Ngày       | Thí nghiệm         | Cấu hình/split                               | Kết quả chính                                | Trạng thái / ghi chú                         |
| :--------- | :----------------- | :------------------------------------------- | :------------------------------------------- | :------------------------------------------- |
| 28/10/2026 | Reproduction Run 1 | Llama-2-7B QLoRA; test 518                   | Accuracy 0,7761; Macro-F1 0,8410             | Baseline reproduction; 1 output invalid      |
| 28/10/2026 | C3                 | Adapter Run 1; test 518; constrained scoring | Accuracy 0,7799; Macro-F1 0,8384; ECE 0,1948 | Hoàn tất; ECE thô, chưa calibration          |
| 29/10/2026 | C1 original        | Qwen2.5-Coder-1.5B; seed 0; split 4.149/462  | Accuracy 0,3052; Macro-F1 0,2611             | Pilot validation                             |
| 29/10/2026 | C1 compact         | Cùng cấu hình/split với original             | Accuracy 0,7900; Macro-F1 0,8560             | Pilot validation; dùng làm C2 diff + summary |
| Chưa chạy  | C2 summary-only    | Cùng model/split/seed/config C1              | Chờ metrics.json                             | Chạy cell C2                                 |
| Chưa chạy  | C2 diff-only       | Cùng model/split/seed/config C1              | Chờ metrics.json                             | Chạy cell C2                                 |

## Quy tắc ghi kết quả mới

Ghi thêm một dòng nhật ký sau mỗi lần chạy: ngày; mã/variant; model và revision nếu có; split và seed; Accuracy, Macro-F1 và F1 từng lớp; ECE và ghi chú calibration; thời gian; đường dẫn artifact; vấn đề hoặc giới hạn. Tách rõ kết quả validation với test. Chỉ mở test chính thức cho cấu hình đã chọn, không dùng test để lựa chọn.

## Tệp tham khảo

- [Báo cáo reproduction](../doc/README_step1.md)
- [Chi tiết hướng cải tiến CHCL](README_HUONG_B_CHCL.md)
- [Cell C1 prompt ablation](KAGGLE_C1_PROMPT_ABLATION.md)
- [Cell C3 constrained scoring](README_HUONG_B_CHCL.md#4-cell-kaggle-ngay-1-c3-tren-adapter-run-1)
- [Audit độ dài token](KAGGLE_STEP1.md)
