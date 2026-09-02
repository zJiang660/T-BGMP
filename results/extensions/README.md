# Camera-ready extension evidence

This directory contains the smallest case-level representation needed to
recompute Tables 4--7 and the final paper's text-only validation results. It
was recovered from the completed formal GPU runs;
raw model responses, scheduler logs, checkpoints, timestamps, and private
machine paths are intentionally excluded.

- `domain_heldout/case_level.csv`: 72 evaluation cases. Random columns are the
  three same-budget controls for each case.
- `frozen_top3/case_level.csv`: 72 unseen-request cases with FP16, aggressive,
  uniform, frozen Top3, Bottom3, and three Random3 outcomes.
- `ruler/screening_case_level.csv`: 600 official-task samples with FP16 and
  aggressive screening outcomes.
- `ruler/recovery_case_level.csv`: all 304 conditional cases, retaining every
  Top1--Top12 result and the predeclared control budgets.
- `weight_sensitivity/rankings.csv`: four complete 36-layer rankings per model.
- `weight_sensitivity/case_level.csv`: first-success evidence for all four
  weight settings. Qwen2.5-3B perturbations reuse the direct equal-weight
  outputs only because every case succeeds within the unchanged Top3 prefix;
  the `evidence` column records this distinction.
- `fixed_layer/case_level.csv`: validated fixed-policy outcomes plus the
  canonical main-ranking exact-at-budget T-BGMP outcomes used by the paper.
- `cross_seed_heldout/`: bidirectional cross-seed Top1--Top12 outcomes and the
  frozen calibration rankings.
- `gemma2_boundary/`: completed Gemma2 reproduction supporting incomplete
  key-only recovery; the Top1--Top12 union restores 18/25 conditional cases.

Run `python scripts/audit_extension_results.py` to reconstruct and verify the
paper-facing tables.
