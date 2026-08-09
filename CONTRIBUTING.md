# Hướng dẫn đóng góp

## Nguyên tắc phạm vi

- Nguồn dữ liệu cuối cùng chỉ gồm TMDb Official API.
- Biến mục tiêu cố định: `revenue >= 2 × budget`.
- Phạm vi mô hình giới hạn ở XGBoost và các đặc trưng có sẵn trước phát hành.
- `popularity`, điểm đánh giá, lượt bình chọn, `revenue`, `profit` và ROI không
  được sử dụng làm đặc trưng dự báo.
- `.env`, token, dữ liệu TMDb và dự đoán cấp từng phim không được đưa vào Git.

## Quy trình đề xuất

1. Tạo nhánh riêng cho thay đổi.
2. Cài đặt các thư viện trong `requirements.txt` vào `.venv`.
3. Duy trì mã nguồn rõ ràng và bổ sung kiểm thử không phụ thuộc dữ liệu riêng tư
   khi phù hợp.
4. Kiểm tra cú pháp và chạy kiểm thử đơn vị trước khi gửi thay đổi.
5. Ghi nhận thí nghiệm mô hình trong `docs/tuning_registry.md`; không ghi đè
   mốc 0,719483 nếu giao thức mới chưa vượt tiêu chí sàng lọc vòng trong.

```powershell
& '.\.venv\Scripts\python.exe' -m compileall -q src tests run_pipeline.py
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Mọi thay đổi về đặc trưng hoặc tối ưu tham số phải giữ nguyên 1.646 định danh
phim, phân hoạch vòng ngoài và quy tắc khớp tiền xử lý trong dữ liệu huấn luyện
của từng vòng kiểm định. Ngoại lệ chỉ áp dụng khi một nghiên cứu mới xác lập
giao thức độc lập và không so sánh trực tiếp với mốc tham chiếu hiện hành.
