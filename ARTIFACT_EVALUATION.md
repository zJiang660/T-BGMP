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

### Claim 3: Ranking Validation

The camera-ready risk ablation, weight sensitivity, cross-seed, and
domain-held-out results are available as canonical cleaned CSVs or reduced
case-level evidence.

How to verify:

```bash
python scripts/build_paper_tables.py
```

Relevant files: `results/paper_tables/table_risk_ablation.csv`,
`results/paper_tables/table_weight_sensitivity.csv`, and
`results/paper_tables/table_domain_heldout.csv`. Cross-seed evidence is under
`results/extensions/cross_seed_heldout/`.

### Claim 4: Frozen and RULER-Style Transfer

The frozen Top3 and RULER-style transfer results are represented by the same
summary rows used in camera-ready Tables 6 and 7.

How to verify:

```bash
python scripts/build_paper_tables.py
python scripts/validate_csv_schema.py
```

Relevant files: `results/paper_tables/table_frozen_top3.csv` and
`results/paper_tables/table_ruler_transfer.csv`.

### Claim 5: Fixed-Layer and Gemma2 Boundary Checks

The fixed-layer comparison combines validated fixed-policy outcomes with the
canonical main-ranking exact-at-budget T-BGMP outcomes. Gemma2-9B is treated
as an incomplete key-only recovery boundary, not as a fifth main model.

How to verify:

```bash
python scripts/audit_extension_results.py
```

Expected output includes:

```text
OBSERVED fixed-layer Qwen3-4B: fixed_l0=54/72, fixed_l0_l7_l25=68/72, tbgmp_top1=54/72, tbgmp_top3=68/72
OBSERVED fixed-layer Llama3.2-3B: fixed_l0=12/25, fixed_l0_l7_l25=15/25, tbgmp_top1=0/25, tbgmp_top3=13/25
OBSERVED Gemma2 cumulative Top12=18/25; unrecovered=7/25
```

### Claim 6: Backend Smoke Path

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
python scripts/check_paper_artifacts.py
python -m pytest tests
```
