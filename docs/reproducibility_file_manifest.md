# Reproducibility File Manifest

The repository is organized around cleaned, paper-ready artifacts.

- `src/tbgmp/`: lightweight analysis helpers used by the demo and scripts.
- `experiments/`: model-free smoke/demo entry points.
- `configs/`: small KV policy JSON files.
- `data/demo/`: cleaned case metadata for inspection.
- `data/schema/`: schema notes for results and policy files.
- `results/paper_tables/`: canonical paper-facing CSVs for Tables 1--7 and
  Figure 2.
- `results/main_evidence/`: sanitized case-level bundles for the four core
  models.
- `results/supporting/`: supporting summaries and the Gemma2 case-level bundle.
- `slurm/xec/`: sanitized XEC Slurm templates.

`tables/paper/`, `figures/paper/`, and `analysis_outputs/` are local generated
directories. They are not part of the tracked artifact and are ignored by Git.

Excluded material includes model weights, HuggingFace or ModelScope caches,
raw cluster logs, large zip archives, credentials, and files containing private
workstation or cluster paths.
