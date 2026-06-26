# Reproducibility Guide

## Reproducibility Levels

- **Level 1:** Cleaned CSV to paper tables and figures. Fully supported.
- **Level 2:** Sanitized case-level outputs to audited paper numbers. Supported
  when the corresponding case-level CSV bundle is present.
- **Level 3:** Rerun model execution through the documented TurboQuant
  integration path. This requires the external `turboquant-pytorch` runtime,
  model weights, a GPU environment, and a backend binding that supports
  arbitrary protected key-layer IDs.

Levels 1 and 2 are directly runnable in this repository. Level 3 provides a
complete experiment-control interface but depends on external model and
quantizer components described in `model_setup.md`.

Backend installation and the current adapter boundary are documented in
`backend_integration.md`; runnable command templates are in
`command_cookbook.md`.

This repository does not include model weights, raw HPC logs, full raw model
responses, or a production quantizer kernel. The minimal demo does not load
language models. Full GPU/model execution requires external model weights and
backend support.

## Environment

Install with either:

```bash
python -m pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate tbgmp-repro
```

## Model-Free Pipeline

Run the compact demonstration:

```bash
python experiments/run_demo_small.py
```

Run the explicit Stage A--F flow:

```bash
python experiments/stage_a_discovery.py
python experiments/stage_b_mine_sensitive_cases.py
python experiments/stage_c_profile_key_risk.py
python experiments/stage_d_topk_recovery.py
python experiments/stage_e_random_bottom_controls.py
python experiments/stage_f_efficiency_analysis.py
```

These commands use `data/demo/` and write small derived files to
`results/audit/`.

## Paper Tables and Figures

```bash
python scripts/build_paper_tables.py
python scripts/build_figures.py
```

Markdown tables are written to `tables/paper/`. The reproduced controls plot is
written to `figures/paper/`.

## Data Verification

```bash
python scripts/validate_csv_schema.py
python scripts/audit_results.py
```

The audit verifies:

- the four-model main set and 183/183 aggregate;
- supporting/main separation;
- Gemma2 value-bottleneck numbers;
- excluded-model classification;
- equality between canonical and grouped CSV copies.
- per-model restoration directly from sanitized case-level Top-k rows;
- source provenance hashes without private source paths.

It writes `results/audit/audit_summary.json` and SHA-256 hashes in
`results/audit/artifact_hashes.csv`.

## Full GPU Reruns

Full model execution is outside this minimal repository because checkpoints,
raw outputs, and the quantized cache runtime are not distributed here.
`experiments/run_full_pipeline.py` supplies discovery, sensitive mining,
Top1--Top12, and same-budget control orchestration through an external backend
contract. The Slurm files are sanitized templates showing the required
interfaces; they are not turn-key institutional jobs.

The included TurboQuant adapter validates the external runtime and then fails
clearly before generation because the upstream first/last-layer protection
interface is not equivalent to T-BGMP's arbitrary key-only layer selection.
No placeholder result is written as a successful model run.

Current limitation: arbitrary risk-ranked protected key-layer ID execution
depends on backend support or the provided patch guide.

Use the dry-run command to validate the orchestration contract without model
execution:

```bash
python experiments/run_full_pipeline.py \
  --dry-run \
  --backend turboquant \
  --model-key qwen25_3b \
  --model-root /path/to/models \
  --turboquant-root /path/to/turboquant-pytorch \
  --output-dir /path/to/outputs/qwen25_3b
```

Use `experiments/smoke_test_backend.py` for a real backend smoke test after the
external runtime has been adapted.
