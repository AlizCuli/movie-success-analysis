# Development Guide

## Environment

- Use Python 3.14 and the project `.venv`.
- Install the pinned dependencies from `requirements.txt`.
- Keep the TMDb Read Access Token in the local `.env` file only.

## Validation

```powershell
& '.\.venv\Scripts\python.exe' -m compileall -q src tests run_pipeline.py
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

The public figure workflow can be checked with:

```powershell
& '.\.venv\Scripts\python.exe' src\generate_report_figures.py
```

The last command requires the local TMDb artifacts described in
[`data/README.md`](data/README.md).

## Experiment protocol

Every new experiment records its hypothesis, new data or features, leakage
controls, inner-validation screening criterion, computational cost, and result
in [`docs/tuning_registry.md`](docs/tuning_registry.md). Do not overwrite the
reference tables or model bundle.

## Benchmark preservation

The reference benchmark is Macro-F1 0.719483 on 1,646 movies using 5 outer × 4
inner nested stratified cross-validation. Preserve the target, split protocol,
feature scope, and out-of-sample evaluation when comparing new work.

## Data and secret handling

Do not commit `.env`, API tokens, raw or processed row-level data, per-movie
predictions, checkpoints, or temporary notebook outputs. Public tables must be
aggregate and contain no movie-level identifiers.
