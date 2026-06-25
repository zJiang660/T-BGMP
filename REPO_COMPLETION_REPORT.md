# Repository Completion Report

Completion date: 2026-06-25

## Current Repository Structure

The repository now contains:

- core analysis modules under `src/tbgmp/`;
- model-free Stage A--F experiments under `experiments/`;
- generic YAML configs and preserved friend-uploaded JSON policies;
- synthetic demo inputs and JSON schemas;
- canonical, grouped, and audited cleaned result tables;
- paper figures, generated Markdown tables, and figure-source notes;
- protocol, interpretation, limitations, reproducibility, and data documents;
- sanitized Slurm templates;
- pip and Conda environment specifications.

The pre-edit inventory is recorded in `repo_structure_before.md`.

## Added Files

Major additions include:

- `src/tbgmp/controls.py`, `retrieval_eval.py`, and `utils.py`;
- Stage A--F model-free experiment scripts;
- five generic YAML config files;
- canonical demo context/cases/expected outputs;
- Stage A/D/E JSON schemas;
- `scripts/audit_results.py` and `scripts/validate_csv_schema.py`;
- generated audit summary and SHA-256 artifact manifest;
- four reproducibility/interpretation documents;
- `environment.yml`;
- a generic A800 Slurm pipeline template;
- Markdown tables generated from the canonical CSVs.

## Modified Files

- README now defines the intended scope, evidence groups, quick start, audit
  commands, and safety boundary.
- Core risk output now includes an explicit rank.
- Policy definitions now include Top-k, Random-k, and Bottom-k categories.
- Metrics include a safe Wilson interval.
- The demo now reads canonical demo CSV data.
- The friend-uploaded replay manifest had one workstation path replaced with a
  redacted placeholder.

All modified pre-existing files were backed up outside the repository before
editing.

## Preserved Friend-Uploaded Content

The following were retained:

- Qwen2.5, Falcon3, and Zephyr policy JSON files;
- grouped main/supporting/boundary result copies;
- paper-table CSV copies;
- replay/evidence metadata after path redaction;
- existing validity and Qwen2.5 control summaries;
- sanitized Slurm files;
- repository fill reports and schema notes.

No valid result CSV was deleted or numerically altered.

## Data Locations

- Main evidence: `results/main_evidence/` and canonical
  `results/paper_tables/table_main_evidence.csv`
- Supporting/control data: `results/supporting/`
- Gemma2 boundary: `results/boundary_excluded/table_gemma2_boundary.csv`
- Excluded models: `results/boundary_excluded/table_boundary_models.csv`
- Integrity outputs: `results/audit/`

## Running the Demo

```bash
python experiments/run_demo_small.py
```

For the explicit model-free stages, run the Stage A--F scripts in filename
order.

## Reproducing Paper Tables and Figures

```bash
python scripts/build_paper_tables.py
python scripts/build_figures.py
```

Markdown tables are written to `tables/paper/`. The controls figure is written
to `figures/paper/`.

## Data Verification

```bash
python scripts/validate_csv_schema.py
python scripts/audit_results.py
```

The audit confirms:

- main evidence model set is exact;
- total sensitive cases are 183;
- total restored cases are 183/183;
- supporting models are not treated as main evidence;
- Gemma2 is boundary-supporting value-bottleneck evidence;
- Gemma2 key-only and K6/V2 are 7/72, while K6/V4 is 72/72;
- excluded models are classified as invalid or interface-limited;
- canonical and grouped CSV copies are equal.

## Safety Check

- Real names: none found
- Affiliations or personal email addresses: none found
- Local workstation paths: none found after redaction
- Private cluster paths: none found
- Credentials or key material: none found
- Model weights: none found
- Raw logs: none found
- Files larger than 25 MB: none found

## Test Results

- `run_demo_small.py`: PASS
- Stage A--F scripts: PASS
- `build_paper_tables.py`: PASS
- `build_figures.py`: PASS
- `validate_csv_schema.py`: PASS
- `audit_results.py`: PASS
- Python compile check: PASS

## Remaining Scope Boundary

The repository does not provide a turn-key full GPU model runner, model
checkpoints, quantizer kernels, or raw cluster outputs. Those components remain
external by design. The included Slurm files are sanitized interface templates,
not directly executable institutional jobs.
