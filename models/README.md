# Gói XGBoost chính thức

| File | Vai trò |
| --- | --- |
| `xgboost_pre_release_operational_bundle.joblib` | Feature builder, preprocessing, XGBoost và threshold |
| `xgboost_pre_release_operational_model.json` | Booster native của XGBoost |
| `xgboost_pre_release_operational_manifest.json` | Cấu hình, phiên bản, checksum và benchmark |

Benchmark ngoài mẫu của gói là Macro-F1 **0,719483** trên 1.646 phim bằng
nested CV 5×4. Model bundle được fit trên toàn bộ dữ liệu sau khi khóa cấu hình,
do đó không có một “test score” riêng cho chính file bundle.

Huấn luyện lại sau khi đã tái tạo dữ liệu local:

```powershell
& '.\.venv\Scripts\python.exe' src\train_final_xgboost.py
```

Xem schema và dự đoán:

```powershell
& '.\.venv\Scripts\python.exe' src\predict_xgboost.py --show-schema
& '.\.venv\Scripts\python.exe' src\predict_xgboost.py input.csv output.csv
```

Schema đầu vào và giới hạn history được mô tả tại
[`docs/model_input_schema.md`](../docs/model_input_schema.md). Không nạp file
`joblib` từ nguồn không tin cậy vì định dạng này có thể thực thi mã khi load.
