# Quick Reproduction

The default quick reproduction path does not require model weights or a GPU.

## 0. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Quick Demo

```bash
python experiments/run_demo_small.py
```

## 2. Audit Paper Numbers

```bash
python scripts/audit_results.py
```

Expected:

```text
Total main: 183/183
```

## 3. Regenerate Paper Tables

```bash
python scripts/build_paper_tables.py
```

## 4. Regenerate Paper Figures

```bash
python scripts/build_figures.py
```

## 5. Validate Schemas

```bash
python scripts/validate_csv_schema.py
```

## 6. Run Tests

```bash
python -m pytest tests
```

## 7. Check Artifact Integrity

```bash
python scripts/check_artifact_integrity.py
```

## 8. Backend Smoke Test

See `docs/smoke_test.md`.

The backend smoke test requires user-supplied model weights, a CUDA GPU, and a
patched or equivalent TurboQuant runtime. It is not part of the default
CPU-only quick path.
