# Version Lock

## Repository

T-BGMP commit: use `git rev-parse HEAD` for the exact artifact revision.

## Python Dependencies

See `requirements.txt`.

## Python Version

Tested with Python >= 3.10.

## External Runtime

TurboQuant PyTorch:

<https://github.com/tonbistudio/turboquant-pytorch>

Known inspected files:

- `turboquant/compressors_v3.py`
- `turboquant/generation_test.py`
- `turboquant/generation_test_v2.py`
- `turboquant/validate_v3.py`

Exact T-BGMP key-only Top-k execution requires arbitrary risk-ranked protected
key-layer ID support. The public TurboQuant interface does not directly expose
this behavior; use the provided patch or an equivalent backend.

## Model IDs

See `configs/model_registry.yaml`.

## Smoke-Test Status

A small backend smoke test has been validated with a patched TurboQuant runtime
and Qwen2.5-3B-Instruct. Sanitized outputs are in `examples/smoke_test/`.
