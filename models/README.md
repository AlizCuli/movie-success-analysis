# Official XGBoost Model Package

| File | Role |
| --- | --- |
| `xgboost_pre_release_operational_bundle.joblib` | Feature builder, preprocessing, XGBoost estimator, and classification threshold |
| `xgboost_pre_release_operational_model.json` | Native XGBoost booster |
| `xgboost_pre_release_operational_manifest.json` | Configuration, versions, checksums, and reference metrics |

The package corresponds to the reference benchmark: Macro-F1 **0.719483** on
1,646 movies under 5×4 nested stratified cross-validation. It was fit on the
full cohort after configuration selection, so the package fit is not an
independent test estimate.

After recreating local data, train the final package with:

```powershell
& '.\.venv\Scripts\python.exe' src\train_final_xgboost.py
```

Inspect the input schema and run inference with:

```powershell
& '.\.venv\Scripts\python.exe' src\predict_xgboost.py --show-schema
& '.\.venv\Scripts\python.exe' src\predict_xgboost.py input.csv output.csv
```

Input requirements and franchise-history limitations are documented in
[`docs/model_input_schema.md`](../docs/model_input_schema.md). Do not load a
`joblib` file from an untrusted source because the format can execute code when
deserialized.
