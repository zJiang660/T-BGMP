# Data Provenance

## Included Data

- cleaned paper-ready CSV files;
- sanitized case-level or compact result files;
- smoke-test sanitized examples;
- generated paper tables and figures derived from the included CSV files.

## Excluded Data

- raw HPC logs;
- full raw model responses;
- model weights;
- local model caches;
- private machine paths;
- private environment files.

## Transformation Pipeline

```text
raw model output JSONL
-> scripts/convert_raw_outputs_to_case_csv.py
-> case-level CSV
-> scripts/audit_results.py
-> paper tables / figures
```

The repository commits only cleaned or sanitized artifacts. The smoke-test
example keeps bounded response excerpts and omits private paths and logs.

## Sensitive-Case Definition

Sensitive cases are selected using retrieval `found` fields, not execution
status:

```text
FP16 found == True and aggressive uniform found == False
```

OOM, invalid FP16 baselines, and incompatible cache interfaces are tracked as
execution conditions rather than retrieval misses.

## Provenance Checks

`scripts/audit_results.py` checks case-level consistency and provenance hashes
for the included compact result bundles. `scripts/check_artifact_integrity.py`
checks repository-level artifact completeness and public-safety constraints.
