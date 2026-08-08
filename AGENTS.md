# Quy tắc làm việc

## Bối cảnh

- Đây là project môn Phân tích dữ liệu về phim ảnh.
- Nguồn dữ liệu chính thức duy nhất của báo cáo và mô hình cuối:
  - TMDb Official API
- Các artifact IMDb cũ chỉ được giữ local để bảo toàn lịch sử pipeline, không
  thuộc phạm vi dữ liệu cuối, không được phân phối qua repository và không được
  dùng trong báo cáo hoặc predictor.

## Quy tắc

1. Luôn giải thích bằng tiếng Việt.
2. Mỗi lần chỉ thực hiện một bước nhỏ.
3. Trước khi viết code, phải giải thích mục đích.
4. Không tự chuyển sang bước tiếp theo.
5. Không tạo dữ liệu giả.
6. Không đưa API key hoặc mật khẩu vào code.
7. Ưu tiên code Python đơn giản, dễ đọc.
8. Không tự commit hoặc push Git nếu chưa được yêu cầu.
9. Không chỉnh sửa trực tiếp dữ liệu gốc.
10. Mỗi thay đổi phải báo rõ file nào đã được tạo hoặc sửa.

## Quy tắc riêng trước khi phát triển mô hình

11. Trước mọi thảo luận, feature engineering hoặc tuning, phải đọc
    `GPT_CONTEXT.md` và `docs/tuning_registry.md`.
12. Không chạy lại một thí nghiệm đã bị bác bỏ nếu chưa nêu rõ giả thuyết, dữ liệu
    hoặc cách biểu diễn mới khác với lần trước.
13. Giữ nguyên benchmark XGBoost pre-release operational Macro-F1 `0.719483` và
    không ghi đè model, bảng hay biểu đồ hiện hành.
14. Mọi thí nghiệm mới phải ghi scope, protocol chống leakage, feature mới, kết
    quả và chênh lệch so với benchmark vào tuning registry.
15. Repository public chỉ giữ `data/README.md` và các file `.gitkeep` trong
    `data/`; không commit dữ liệu gốc, trung gian hoặc đã xử lý.
