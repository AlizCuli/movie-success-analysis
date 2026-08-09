# Hướng dẫn đóng góp

## Nguyên tắc phạm vi

- Nguồn dữ liệu cuối chỉ là TMDb Official API.
- Target cố định: `revenue >= 2 × budget`.
- Chỉ phát triển XGBoost với predictor có sẵn trước phát hành.
- Không dùng popularity, rating, vote, revenue, profit hoặc ROI làm predictor.
- Không commit `.env`, token, dữ liệu TMDb hoặc dự đoán cấp từng phim.

## Quy trình đề xuất

1. Tạo branch cho thay đổi.
2. Cài dependency từ `requirements.txt` trong `.venv`.
3. Viết code đơn giản, thêm/điều chỉnh test không cần dữ liệu mật khi có thể.
4. Chạy compile và unit test trước khi gửi thay đổi.
5. Ghi thí nghiệm mô hình vào `docs/tuning_registry.md`; không ghi đè benchmark
   0,719483 nếu protocol mới chưa vượt qua inner gate hợp lệ.

```powershell
& '.\.venv\Scripts\python.exe' -m compileall -q src tests run_pipeline.py
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Mọi thay đổi feature/tuning phải giữ nguyên 1.646 ID, outer split và quy tắc fit
preprocessing bên trong training/inner CV, trừ khi nghiên cứu mới nêu rõ một
protocol độc lập và không so sánh trực tiếp với benchmark cũ.

