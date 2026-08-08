# Gói mô hình XGBoost chính thức

Thư mục này lưu ba artifact của mô hình XGBoost pre-release operational:

- `xgboost_pre_release_operational_bundle.joblib`: bộ xây dựng đặc trưng, bộ
  tiền xử lý, mô hình XGBoost và ngưỡng phân loại;
- `xgboost_pre_release_operational_model.json`: mô hình ở định dạng native của
  XGBoost;
- `xgboost_pre_release_operational_manifest.json`: cấu hình, phiên bản thư viện,
  checksum dữ liệu và kết quả benchmark ngoài mẫu.

Lệnh tái tạo gói mô hình trên máy đã có dữ liệu local:

```powershell
& '.\.venv\Scripts\python.exe' 'src\train_final_xgboost.py'
```

Macro-F1 `0,719483` là kết quả nested outer-OOF trên 1.646 phim, không phải điểm
huấn luyện của mô hình được xây dựng trên toàn bộ dữ liệu. Các tệp dữ liệu huấn
luyện không được đóng gói trong thư mục này và không được phân phối qua
repository.
