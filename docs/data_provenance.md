# Data Provenance

## Included Data

- cleaned paper-ready CSV files;
- sanitized case-level or compact result files;
- smoke-test sanitized examples;
- scripts that regenerate paper tables and Figure 2 from the included CSV files.

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

## Recovered Full-Layer Profiles

The Qwen2.5-14B and Llama3.2-3B `risk_ranking.csv` files are recovered from
the original HPC Stage C outputs rather than reconstructed from protected-layer
lists. They contain one row per Transformer layer, all three empirical risk
components, the score, and the exact rank used by the corresponding recovery
sweep. Source filenames, byte sizes, and SHA-256 hashes are recorded in each
model directory's `source_provenance.json`.

The Llama3.2-3B run used the normalized Full score implemented by the current
pipeline. The historical Qwen2.5-14B runner used the same three components but
summed the raw MSE, log-IP, and inverse-effective-dimension values before
ranking. Its `score_protocol` field and provenance note preserve that distinction
explicitly; reproducing the reported Qwen2.5-14B rows requires that recorded
historical rank, while a new experiment should use the current normalized score.
