# Camera-ready extension experiments

The formal held-out, frozen-policy, RULER-transfer, and weight-sensitivity
artifacts were recovered from the completed GPU projects rather than recreated
from paper summaries. Their reduced case-level outputs live under
`results/extensions/` and are checked by:

```bash
python scripts/audit_extension_results.py
```

## Domain-held-out ranking

`experiments/extensions/run_domain_heldout.py` is the formal runner with all
machine paths converted to arguments. It profiles Math and Literature, freezes
the ranking, and evaluates Science and Code. The input classification can be
the corresponding model's `results/main_evidence/*/sensitive_cases.csv`.

```bash
python experiments/extensions/run_domain_heldout.py \
  --turboquant-root "$TURBOQUANT_ROOT" \
  --model-id qwen3_4b \
  --model-path "$MODEL_ROOT/Qwen3-4B-Instruct-2507" \
  --case-classification results/main_evidence/qwen3_4b/sensitive_cases.csv \
  --output-dir "$OUTPUT_ROOT/domain_heldout/qwen3_4b" \
  --output-prefix qwen3_4b
```

The runtime root must expose the formal helper module
`run_tbgmp_single_model_hpc.py`; this is the same helper API used by the
completed run. The experiment settings are recorded in
`configs/extensions/domain_heldout.yaml`.

The held-out runner profiles the frozen domains with the same Full-score
implementation as the main method: TurboQuant `MSECompressor`, bit-normalized
95th-percentile MSE and inner-product distortion, effective dimension, and
per-model min-max normalization. Its ranking CSV is tagged
`paper_full_normalized_v1`; the runner refuses to resume from an older output
directory containing a different scoring protocol.

## Frozen Top3 policy

This experiment selects one model-level Top3 policy from calibration results
and applies it once per unseen request. It does not perform a per-request
Top-k sweep. Set the runtime paths referenced by
`configs/extensions/frozen_top3.json`, then run one row from the task manifest:

```bash
export PROJECT_ROOT="$PWD"
export TBGMP_FROZEN_ROOT="$OUTPUT_ROOT/frozen_top3"
export TBGMP_BACKEND_SCRIPT="$TURBOQUANT_BACKEND_SCRIPT"
python experiments/extensions/run_frozen_top3.py --task-id 0
```

The exact evaluation grid is retained in
`data/extension_cases/frozen_top3_cases.json`; the ten policies are listed in
`configs/extensions/frozen_top3_tasks.csv`.

## RULER task transfer

The RULER configuration pins the upstream repository and generator commits,
three official tasks, two lengths, and 50 samples per task-length stratum.
Generate `RULER_FORMAL_CASE_MANIFEST.json` with those pinned official tools and
place it under `$TBGMP_RULER_ROOT/manifests/`. Then run screening, build the
conditional set, and run the frozen ranking controls:

```bash
export TBGMP_RULER_ROOT="$OUTPUT_ROOT/ruler"
python experiments/extensions/run_ruler_transfer.py \
  --stage screening --model qwen3_4b --task niah_multikey_1 --length 4096
python experiments/extensions/build_ruler_conditional_sets.py
python experiments/extensions/run_ruler_transfer.py \
  --stage top --model qwen3_4b --chunk-index 0 --total-chunks 10
```

Repeat the declared task/length grid and the `top`, `bottom`, and `random`
stages for both models. The runner is checkpointed and rejects duplicate rows.
The official scorer is the case-insensitive all-reference containment metric
used by the pinned RULER interface.

## Weight sensitivity

This is an offline re-ranking analysis over completed GPU outputs, so it does
not require a new model run. The Qwen3 outputs directly cover all four weight
settings. For Qwen2.5-3B, all settings share the same Top3 prefix and all 72
cases already recover within that prefix; later budgets are not represented as
new inference. Rebuild the two reduced files with:

```bash
python experiments/extensions/build_weight_sensitivity.py \
  --qwen3-profile /path/to/qwen3_profile.csv \
  --qwen25-profile /path/to/qwen25_profile.csv \
  --qwen3-details /path/to/qwen3_weight_sweep.csv \
  --qwen25-full-cases results/main_evidence/qwen25_3b/risk_ablation_first_success_by_case.csv \
  --output-dir /path/to/rebuilt/weight_sensitivity
```

`results/extensions/source_provenance.json` records SHA-256 identifiers for
the formal source files and shard manifests without exposing host paths.
