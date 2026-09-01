# Cleaned Results Schema

The cleaned CSV files under `results/` are derived from paper-ready summaries
or sanitized case-level outputs, not raw cluster logs. `tables/paper/` is a
generated Markdown output directory and is not a canonical data source.

## Common fields

- `model`: display name used in the manuscript tables.
- `model_id`: stable short identifier where available.
- `sensitive`: number of FP16-pass/aggressive-fail cases.
- `recovered`: recovered sensitive cases, formatted as `n/N` where used.
- `first_success_k`: first Top-k key-protection budget that restored the case.
- `kv_saving_percent`: reported KV-cache saving relative to FP16.
- `status`: execution validity category after cleaning.
- `found`: retrieval success flag; OOM and invalid baselines are not retrieval
  misses.

## Cleaning rules

- Model weights and model-cache paths are excluded.
- Raw logs and full generations are excluded.
- Private workstation and HPC paths are excluded or replaced by placeholders.
- Boundary and excluded models are separated from main evidence.
