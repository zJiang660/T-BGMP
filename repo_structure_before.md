# Repository Structure Before Completion

Audit date: 2026-06-25

## Existing Directories

- `src/tbgmp/`: minimal risk, policy, sensitive-case, and metric utilities.
- `experiments/`: one model-free demonstration.
- `configs/`: friend-uploaded Qwen2.5, Falcon3, and Zephyr policy JSON files.
- `data/demo/`: friend-uploaded replay and evidence-candidate metadata.
- `data/schema/`: policy schema and cleaned-result notes.
- `results/paper_tables/`: canonical cleaned paper-ready CSV files.
- `results/main_evidence/`, `results/supporting/`,
  `results/boundary_excluded/`: friend-uploaded grouped copies.
- `figures/paper/`: manuscript figures and one reproduced controls plot.
- `tables/paper/`: friend-uploaded paper-table CSV copies.
- `docs/`: protocol and supporting reproducibility notes.
- `slurm/xec/`: sanitized cluster templates.

## Existing Core Code

- Sensitive-case selection correctly uses retrieval `found` fields.
- Risk scoring implements normalized MSE, log inner-product distortion, and
  inverse effective dimension.
- Policy utilities provide uniform policies and Top-k construction.
- Metrics provide recovery rate and KV-saving advantage.

## Existing Results and Figures

- Seven canonical CSV files are present in `results/paper_tables/`.
- Grouped main, supporting, and boundary copies are present.
- Five paper PDF figures and one regenerated PNG are present.

## Missing or Incomplete Items

- Risk output lacks an explicit `rank` column.
- Random-k control utilities and answer matching are missing.
- Stage A--F model-free pipeline scripts are missing.
- YAML experiment/model/prompt configs are missing.
- Demo context, canonical demo cases, expected outputs, and Stage A/D/E JSON
  schemas are missing.
- Result audit and CSV schema validation scripts are missing.
- Reproducibility, interpretation, limitations, and data-description documents
  are missing.
- `environment.yml`, audit outputs, figure-source notes, and a generic Slurm
  template are missing.
- README does not yet expose the audit commands and evidence taxonomy in enough
  detail.

## Files Requiring Attention

- `data/demo/old_replay_case_manifest.json` contains a local workstation path
  in its metadata root and must be anonymized.
- Existing Slurm files are sanitized but reference external runner scripts that
  are not part of this minimal repository; they should be documented as
  templates rather than executable end-to-end assets.

## Content Preserved

All friend-uploaded configs, cleaned result copies, table copies, summaries,
schema notes, and sanitized Slurm files are retained. No valid result file is
deleted or overwritten.
