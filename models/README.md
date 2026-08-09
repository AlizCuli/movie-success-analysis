# Gói XGBoost chính thức

| File | Vai trò |
| --- | --- |
| `xgboost_pre_release_operational_bundle.joblib` | Bộ kiến tạo đặc trưng, tiền xử lý, XGBoost và ngưỡng phân loại |
| `xgboost_pre_release_operational_model.json` | Booster native của XGBoost |
| `xgboost_pre_release_operational_manifest.json` | Cấu hình, phiên bản, checksum và chỉ số tham chiếu |

Kết quả ngoài mẫu tham chiếu của gói là Macro-F1 **0,719483** trên 1.646 phim,
được ước lượng bằng kiểm định chéo phân tầng lồng nhau 5×4. Gói mô hình được
khớp trên toàn bộ dữ liệu sau khi khóa cấu hình; do đó, bản thân file đóng gói
không có một chỉ số kiểm thử độc lập riêng.

Huấn luyện lại sau khi đã tái tạo dữ liệu cục bộ:

```powershell
& '.\.venv\Scripts\python.exe' src\train_final_xgboost.py
```

Xem lược đồ và thực hiện dự đoán:

```powershell
& '.\.venv\Scripts\python.exe' src\predict_xgboost.py --show-schema
& '.\.venv\Scripts\python.exe' src\predict_xgboost.py input.csv output.csv
```

Lược đồ đầu vào và giới hạn của đặc trưng lịch sử được mô tả tại
[`docs/model_input_schema.md`](../docs/model_input_schema.md). Không nạp file
`joblib` từ nguồn không tin cậy vì định dạng này có thể thực thi mã khi được
nạp.
