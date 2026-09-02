# SLURM Launch Templates

The two `submit_full_pipeline_a800_template.sbatch` files are the canonical
cluster entry points. They call the same resumable Stage A--F runner and differ
only in the cluster label, allowing scheduler directives to be adapted without
changing the experiment command.

Before submission, export these site-local values:

```bash
export TBGMP_ROOT=/path/to/T-BGMP
export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
export CASES_FILE=/path/to/cases.csv
export RISK_RANKING=/path/to/risk_ranking.csv
export MODEL_KEY=qwen25_3b
```

Set valid `--partition` and `--account` directives in the copied template, then
submit it from a directory where the scheduler can create its stdout/stderr
files. The runner writes durable JSONL/CSV checkpoints and a provenance
manifest under `OUTPUT_ROOT`; submitting the same command again resumes only
incomplete identities.

The older stage-named launchers were removed because they called the full
pipeline despite their names. `submit_smoke_test_a800_template.sbatch` remains
an optional, separate backend check and is not part of a formal experiment run.
