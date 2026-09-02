# Version Lock

## Repository

T-BGMP commit: use `git rev-parse HEAD` for the exact artifact revision.

## Python Dependencies

The CPU-only analysis path uses `requirements.txt`. Full GPU execution has two
recorded, exact profiles because completed XEC and SIP runs used different
Python environments:

- `requirements-gpu-xec.txt`
- `requirements-gpu-sip.txt`

Package versions, Python versions, CUDA provenance, model fingerprints, and the
TurboQuant base commit are machine-readable in `configs/runtime_lock.yaml`.
Validate a prepared environment before a rerun with:

```bash
python scripts/check_runtime_lock.py --profile xec_gpu \
  --turboquant-root /path/to/turboquant-pytorch \
  --model-root /path/to/models --model-key qwen3_4b
```

## Python Version

The recorded profiles use Python 3.10.20 (XEC) and Python 3.12.12 (SIP).

## External Runtime

TurboQuant PyTorch:

<https://github.com/tonbistudio/turboquant-pytorch>

Inspected upstream commit:

`999713889a18c0ffa20c62a65e7cbbe5746794e3`

Known inspected files:

- `turboquant/compressors_v3.py`
- `turboquant/generation_test.py`
- `turboquant/generation_test_v2.py`
- `turboquant/validate_v3.py`

Exact T-BGMP key-only Top-k execution requires arbitrary risk-ranked protected
key-layer ID support. The public TurboQuant interface does not directly expose
this behavior; use the provided patch or an equivalent backend.

## Model IDs

Human-readable IDs are in `configs/model_registry.yaml`; immutable snapshot
revisions where retained and content fingerprints for the actual paper
checkpoints are in `configs/runtime_lock.yaml`. For historical ModelScope
downloads that recorded only a moving `master` label, the config and
weight-index SHA-256 values are the authoritative identity. This avoids
inventing a repository commit after the fact.

## Smoke-Test Status

A small backend smoke test has been validated with a patched TurboQuant runtime
and Qwen2.5-3B-Instruct. Sanitized outputs are in `examples/smoke_test/`.
