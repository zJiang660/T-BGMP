# Data Description

## Canonical Paper Tables

`results/paper_tables/` is the canonical cleaned result source:

- `table_main_evidence.csv`: four main models and matched controls.
- `table_first_success_k.csv`: first-success Top-k budget summaries.
- `table_control_statistics.csv`: Top/Random/Bottom rates and tests.
- `table_qwen25_scale.csv`: within-family scale comparison.
- `table_supporting_models.csv`: supporting and boundary-supporting outcomes.
- `table_gemma2_boundary.csv`: value-bottleneck policy contrast.
- `table_boundary_models.csv`: invalid or interface-limited executions.

## Grouped Copies

`results/main_evidence/`, `results/supporting/`, and
`results/boundary_excluded/` contain friend-uploaded grouped copies. The audit
script confirms that they match the canonical tables.

## Sanitized Case-Level Evidence

The model subdirectories under `results/main_evidence/` contain:

- `stage_a_discovery.csv`
- `sensitive_cases.csv`
- `risk_ranking.csv`
- `topk_recovery.csv`
- `random_bottom_controls.csv`
- `first_success_cases.csv`
- `efficiency_summary.csv`
- `source_provenance.json`

The Gemma2 bundle follows the same structure under
`results/supporting/gemma2_9b/`. Responses are truncated to short excerpts;
host, device, partition, timestamps, paths, and raw logs are excluded.

Each provenance file records source filenames, byte sizes, and SHA-256 hashes
without recording the private source location.

## Demo Data

`data/demo/demo_cases.csv` is synthetic and intentionally tiny. It demonstrates
FP16-valid sensitive, tolerant, and invalid cases. The older replay/evidence
metadata files are cleaned review artifacts; they are not full raw generations.

## Excluded Data

The repository excludes model weights, model caches, raw cluster logs, complete
responses, private paths, credentials, and large experiment outputs.

## Integrity

Run `python scripts/audit_results.py` to regenerate a SHA-256 manifest for all
cleaned result CSVs and to verify the key numerical claims.
