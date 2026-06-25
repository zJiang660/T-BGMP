# Reproducibility File Manifest

The repository is organized around cleaned, paper-ready artifacts.

- `src/tbgmp/`: lightweight analysis helpers used by the demo and scripts.
- `experiments/`: model-free smoke/demo entry points.
- `configs/`: small KV policy JSON files.
- `data/demo/`: cleaned case metadata for inspection.
- `data/schema/`: schema notes for results and policy files.
- `results/main_evidence/`: cleaned main evidence CSVs.
- `results/supporting/`: cleaned supporting and control CSVs.
- `results/boundary_excluded/`: boundary and excluded-model CSVs.
- `figures/paper/`: paper figures and reproduced control plot.
- `tables/paper/`: paper-ready CSV table copies.
- `slurm/xec/`: sanitized XEC Slurm templates.

Excluded material includes model weights, HuggingFace or ModelScope caches,
raw cluster logs, large zip archives, credentials, and files containing private
workstation or cluster paths.
