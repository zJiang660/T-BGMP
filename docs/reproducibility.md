# Reproducibility Guide

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

It writes `results/audit/audit_summary.json` and SHA-256 hashes in
`results/audit/artifact_hashes.csv`.

## Full GPU Reruns

Full model execution is outside this minimal repository because checkpoints,
raw outputs, and the quantized cache runtime are not distributed here. The
Slurm files are sanitized templates showing the required external interfaces;
they are not turn-key cluster jobs.
