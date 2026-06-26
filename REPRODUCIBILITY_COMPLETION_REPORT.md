# Reproducibility Completion Report

## Current Capability

This repository supports Level 1 and Level 2 reproducibility. It now provides
a documented Level 3 integration path using the external TurboQuant PyTorch
runtime. Level 3 rerunning still requires user-supplied model weights, GPU
hardware, and local backend configuration.

- Level 1 rebuilds paper tables and figures from cleaned CSV files.
- Level 2 audits the main paper numbers from sanitized case-level files.
- Level 3 provides configs, model registration, command-line runners, a
  TurboQuant validation adapter, raw-output conversion, and sanitized Slurm
  templates.

The repository does not provide model checkpoints, raw HPC logs, complete raw
responses, or a production quantizer kernel. It therefore does not claim
one-command reproduction of all GPU experiments from a blank machine.

## Case-Level Data

Complete compact audit bundles are present for:

- `results/main_evidence/qwen3_4b/`
- `results/main_evidence/qwen25_3b/`
- `results/main_evidence/qwen25_14b/`
- `results/main_evidence/llama32_3b/`
- `results/supporting/gemma2_9b/`

These bundles retain retrieval `found` fields, policies, precision settings,
protected layers, context metadata, compact response excerpts, and provenance
hashes while excluding private paths and raw logs.

The supporting Mistral, Yi, Zephyr, and SmolLM2 directories and the excluded
Gemma3, Qwen3.5, InternLM, and GLM directories contain explicit
`MISSING_CASE_LEVEL_DATA.md` declarations. Their paper-ready summaries remain
available, but full sanitized case-level outputs are not claimed.

## Runner Support

The model-free Stage A--F scripts expose command-line help and configurable
inputs/outputs:

- `experiments/stage_a_discovery.py`
- `experiments/stage_b_mine_sensitive_cases.py`
- `experiments/stage_c_profile_key_risk.py`
- `experiments/stage_d_topk_recovery.py`
- `experiments/stage_e_random_bottom_controls.py`
- `experiments/stage_f_efficiency_analysis.py`

The demo path validates response matching, sensitive-case selection, risk
ranking, Top-k policy construction, matched controls, and KV-saving analysis
without loading a language model.

`experiments/run_full_pipeline.py` provides the model-execution orchestration
contract. It accepts `--backend turboquant`, `--turboquant-root`,
`--model-key`, `--model-root`, and `--output-dir`, while retaining support for
custom `module.path:factory` backends.

The external TurboQuant source is not copied or added as a submodule. The
adapter validates the configured runtime and confirms that `TurboQuantV3` is
available. It then stops before generation until the upstream cache interface
is locally adapted for arbitrary protected key-layer IDs with unchanged value
precision. The published first/last-layer count is not treated as an
equivalent implementation.

## TurboQuant API Finding

- Repository checked: <https://github.com/tonbistudio/turboquant-pytorch>
- Files inspected: `turboquant/compressors_v3.py`,
  `turboquant/generation_test.py`, `turboquant/generation_test_v2.py`, and
  `turboquant/validate_v3.py`.
- `protected_layers` semantics: prefix/suffix layer count.
- Arbitrary protected key-layer ID support: no direct public binding found.
- Key-only protection support: no direct public binding found.
- `residual_window` support: present in the inspected V3 compressor/cache path.
- Patch needed: yes, for exact T-BGMP execution.

Detailed findings are in `docs/turboquant_api_findings.md`.

## Backend Adapter Status

- Base interface: `src/tbgmp/backends/base.py` defines request/result objects
  and the backend protocol.
- TurboQuant adapter: `src/tbgmp/backends/turboquant_backend.py` validates the
  external runtime and exposes `check_available()`.
- Real generate binding: not claimed. The adapter raises a clear error until
  arbitrary risk-ranked protected key-layer IDs are bound.
- Patch guide: `docs/turboquant_patch_guide.md`.
- Proposed patch sketch:
  `patches/turboquant_arbitrary_protected_layers.patch`.

## Smoke Test Status

`experiments/smoke_test_backend.py` provides a real backend smoke-test entry.
It fails before writing raw output when the backend is not bound to generation.
Real backend smoke testing has not been completed in this repository because
the exact arbitrary key-layer binding is still required.

## Raw-Output Conversion

`scripts/convert_raw_outputs_to_case_csv.py` converts raw JSONL into compact
case-level CSV. It recomputes `found`, removes complete responses and metadata,
and retains only a bounded response excerpt. The input contract is documented
in `docs/raw_output_schema.md` and
`data/schema/raw_output_schema.json`.

## Audit Results

`python scripts/audit_results.py` completed successfully using case-level CSV:

```text
Qwen3-4B: 72/72
Qwen2.5-3B: 72/72
Qwen2.5-14B: 14/14
Llama3.2-3B: 25/25
Total main: 183/183

Gemma2 key-only Top-k: 7/72
Gemma2 Uniform K6/V2: 7/72
Gemma2 Uniform K6/V4: 72/72
```

The audit also verifies that sensitive cases use retrieval `found` fields,
supporting models are excluded from the main total, excluded models are not
counted as retrieval failures, grouped summaries match canonical tables, and
provenance hashes are present. A mismatch produces a FAIL and nonzero exit.

## Local Test Results

- `python experiments/run_demo_small.py`: PASS
- Stage A--F model-free sequence: PASS
- `python scripts/audit_results.py`: PASS
- `python scripts/build_paper_tables.py`: PASS
- `python scripts/build_figures.py`: PASS
- `python scripts/validate_csv_schema.py`: PASS
- `python experiments/run_full_pipeline.py --help`: PASS
- `python experiments/run_full_pipeline.py --dry-run --backend turboquant
  --model-key qwen25_3b --model-root /path/to/models --turboquant-root
  /path/to/turboquant-pytorch --output-dir /path/to/outputs`: PASS
- `python experiments/stage_a_discovery.py --help`: PASS
- `python scripts/convert_raw_outputs_to_case_csv.py --help`: PASS
- `python experiments/smoke_test_backend.py --help`: PASS
- Synthetic raw JSONL to case-level CSV conversion: PASS
- Pytest suite: PASS
- TurboQuant unavailable/incomplete integration failure path: PASS
- Python compilation check for `experiments`, `scripts`, and `src`: PASS

The full GPU pipeline was not executed. The adapter's exact T-BGMP policy
binding is intentionally incomplete and reports this before writing a model
result. No success was fabricated for that level.

## Safety and Size Checks

- Model weights or checkpoint-like files: none found.
- Raw cluster logs or model caches: none found.
- Raw generation data: only the one-line synthetic conversion fixture under
  `data/demo/`; no real model response bundle is included.
- Real local absolute paths: none found.
- Real HPC paths: none found.
- Usernames, institutional identifiers, email addresses, or credentials: none
  found.
- Files larger than 25 MB: none found.
- Largest repository file at the time of inspection: below 0.2 MB.

## Remaining Limitations

- Supporting and excluded/boundary models other than Gemma2 do not yet have
  complete sanitized case-level bundles in this repository.
- Full model inference depends on model-specific cache APIs and an external
  TurboQuant-compatible backend.
- The current upstream protected-layer interface uses first/last layer counts;
  it still needs a local extension for arbitrary risk-ranked key-layer IDs.
- The included patch is a proposed guide, not a validated upstream patch.
- The repository does not implement or benchmark a production quantizer
  kernel and does not claim deployment speedup.
- The reported evidence remains conditioned on FP16-pass/aggressive-fail exact
  retrieval cases rather than general long-context capability.
