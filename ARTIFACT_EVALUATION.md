# Artifact Evaluation Guide

This repository supports paper-table and figure reproduction, case-level
result auditing, and a validated small-scale TurboQuant backend smoke path.
Full paper-scale GPU reruns require user-supplied model weights, GPU
resources, and a patched or equivalent TurboQuant runtime.

## Claims Supported by This Artifact

### Claim 1: Main Conditional Recovery

The four main models recover 183/183 conditional sensitive cases under the
T-BGMP diagnostic recovery protocol.

How to verify:

```bash
python scripts/audit_results.py
```

Expected output:

```text
Qwen3-4B: 72/72
Qwen2.5-3B: 72/72
Qwen2.5-14B: 14/14
Llama3.2-3B: 25/25
Total main: 183/183
```

### Claim 2: Risk-Guided Protection Beats Same-Budget Controls

Top-risk key-layer protection is more reliable than random or bottom-risk
layer protection at the same protection budget on the main evidence cases.

How to verify:

```bash
python scripts/audit_results.py
python scripts/build_paper_tables.py
```

Relevant file: `results/paper_tables/table_control_statistics.csv`.

### Claim 3: Qwen2.5 Scale Comparison

The Qwen2.5-3B and Qwen2.5-14B results are reported as a scale comparison
within the conditional failure-recovery setting.

How to verify:

```bash
python scripts/build_paper_tables.py
```

Relevant file: `results/paper_tables/table_qwen25_scale.csv`.

### Claim 4: Gemma2 Boundary Case

Gemma2-9B is treated as a value-bottleneck boundary-supporting case, not as a
fifth main evidence model.

How to verify:

```bash
python scripts/audit_results.py
```

Expected output includes:

```text
Gemma2 key-only Top-k: 7/72
Gemma2 Uniform K6/V2: 7/72
Gemma2 Uniform K6/V4: 72/72
```

### Claim 5: Backend Smoke Path

The pipeline can run a small TurboQuant backend smoke test with a patched or
equivalent backend.

How to verify:

```bash
python experiments/smoke_test_backend.py --help
```

See `docs/smoke_test.md` and `examples/smoke_test/` for the validated XEC
smoke example.

## Reproducibility Levels

### Level 1: Cleaned CSV to Paper Tables and Figures

- Estimated time: under 5 minutes.
- Hardware: CPU only.
- Commands:

```bash
python scripts/build_paper_tables.py
python scripts/build_figures.py
```

### Level 2: Case-Level or Paper-Ready CSV to Audit Numbers

- Estimated time: under 5 minutes.
- Hardware: CPU only.
- Command:

```bash
python scripts/audit_results.py
```

### Level 3: Small Backend Smoke Test

- Estimated time: minutes for a small model smoke test, depending on GPU and
  model loading time.
- Hardware: CUDA GPU required.
- Dependencies: user-supplied model weights and patched or equivalent
  TurboQuant runtime.
- Entry point: `docs/smoke_test.md`.

### Level 4: Full Paper-Scale GPU Rerun

This is not packaged as a one-command artifact. It requires user-supplied
model weights, GPU resources, and a patched or equivalent TurboQuant runtime.
The repository provides scripts, configs, documentation, and Slurm templates to
prepare such reruns.

## What Is Not Included

- model weights;
- raw HPC logs;
- full raw model responses;
- production quantizer kernels;
- full paper-scale raw outputs;
- private machine paths;
- private environment files or credentials.

## Recommended Evaluation Order

```bash
python experiments/run_demo_small.py
python scripts/audit_results.py
python scripts/build_paper_tables.py
python scripts/build_figures.py
python scripts/validate_csv_schema.py
python scripts/check_artifact_integrity.py
python -m pytest tests
```
