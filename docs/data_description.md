# Data Description

## Canonical Paper Tables

`results/paper_tables/` is the canonical cleaned result source:

- `table_main_evidence.csv`: four main models and matched controls.
- `table_first_success_k.csv`: first-success Top-k budget summaries.
- `table_control_statistics.csv`: camera-ready Table 1 rates and bootstrap
  confidence intervals.
- `figure_2_domain_recovery.csv`: camera-ready Figure 2 curve values.
- `table_risk_ablation.csv`: camera-ready Table 3.
- `table_weight_sensitivity.csv`: camera-ready Table 4.
- `table_domain_heldout.csv`: camera-ready Table 5.
- `table_frozen_top3.csv`: camera-ready Table 6.
- `table_ruler_transfer.csv`: camera-ready Table 7.
- `table_qwen25_scale.csv`: supplementary within-family scale summary.
- `table_supporting_models.csv`: supporting and boundary-supporting outcomes.
- `table_gemma2_boundary.csv`: value-bottleneck policy contrast.
- `table_boundary_models.csv`: invalid or interface-limited executions.

Paper-facing summaries exist only in `results/paper_tables/`. Other result
directories contain case-level bundles or supporting material and do not
duplicate the canonical summary CSVs.

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

For Qwen2.5-3B, the original main-recovery bundle and the later camera-ready
Full-ranking analysis are both retained without conflating their protocols.
The current Table 2 and Table 3 source files are
`risk_ablation_first_success_by_case.csv`, `risk_ablation_topk_curve.csv`, and
`risk_ablation_summary.csv`. The superseded derived `first_success_cases.csv`
was removed from that model directory.

The Gemma2 bundle follows the same structure under
`results/supporting/gemma2_9b/`. Responses are truncated to short excerpts;
host, device, partition, timestamps, paths, and raw logs are excluded.

Models without released case-level bundles are listed once in
`docs/missing_case_level_data.md`.

Each provenance file records source filenames, byte sizes, and SHA-256 hashes
without recording the private source location.

## Demo Data

`data/demo/demo_cases.csv` is synthetic and intentionally tiny. It demonstrates
FP16-valid sensitive, tolerant, and invalid cases. The older replay/evidence
metadata files are cleaned review artifacts; they are not full raw generations.

## Excluded Data

The repository excludes model weights, model caches, raw cluster logs, complete
responses, private paths, credentials, and large experiment outputs.

## Common Fields

- `found`: whether the expected answer string was retrieved from the response;
  this field defines sensitive cases and must not be replaced by `status`.
- `status`: execution outcome such as success, error, invalid baseline, or OOM.
- `policy`: evaluated precision/protection configuration.
- `key_bits` / `value_bits`: default cache precision for keys and values.
- `residual_window`: recent-token window retained by the backend policy.
- `first_success_k`: smallest Top-k protection budget that restored retrieval.
- `protected_layers`: key-layer indices receiving protected precision.
- `kv_saving`: reported or estimated KV-cache saving relative to FP16.

## Integrity

Run `python scripts/audit_results.py` to regenerate a SHA-256 manifest for all
cleaned result CSVs and to verify the key numerical claims.
