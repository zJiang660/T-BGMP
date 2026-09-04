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

## Diagnostic and Ablation Rankings

The main recovery sweeps and the Frozen Top3/RULER extensions must use the
exact ranking that was active when their model outputs were generated. For
Qwen3-4B this is the normalized Full score from the formal SIP profile. For
Qwen2.5-3B the completed diagnostic run used the historical raw-component
score `MSE + log1p(IP) + 1/d_eff`; its exact policy order is preserved in
`results/main_evidence/qwen25_3b/risk_ranking.csv`.

The later camera-ready risk-ablation run recomputed normalized MSE-only,
MSE+IP, and Full rankings. Those outputs support Tables 3 and 4 and are kept
under the dedicated risk-ablation and weight-sensitivity artifacts. They do
not retroactively change the layer order used by the already completed main,
Frozen Top3, or RULER executions. Tests enforce that every diagnostic policy
prefix and extension config matches its corresponding authoritative ranking
file and SHA-256 hash.
