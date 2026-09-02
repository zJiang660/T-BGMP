# T-BGMP Artifact

This repository contains the reproducibility artifact for **T-BGMP: An
Empirical Characterization of Key-Layer Protection for KV-Cache Quantization
Failure Recovery**.

The artifact supports:

1. reproducing paper tables and figures from cleaned CSV files;
2. auditing main paper numbers from sanitized result files;
3. running a small validated TurboQuant backend smoke path;
4. preparing user-side full reruns with model weights, GPU resources, and a
   patched or equivalent TurboQuant runtime.

This repository does not redistribute model weights, raw HPC logs, full raw
model responses, or production quantizer kernels.

## Artifact Entry Points

- [`REPRODUCE.md`](REPRODUCE.md): shortest CPU-only reproduction path.
- [`ARTIFACT_EVALUATION.md`](ARTIFACT_EVALUATION.md): evaluator-oriented
  guide, expected outputs, and reproducibility levels.
- [`CLAIMS_TO_ARTIFACTS.md`](CLAIMS_TO_ARTIFACTS.md): paper claims mapped to
  files and commands.
- [`PAPER_RESULTS_MANIFEST.yaml`](PAPER_RESULTS_MANIFEST.yaml): machine-readable
  result manifest.
- [`docs/model_setup.md`](docs/model_setup.md): model download and path setup.
- [`docs/backend_integration.md`](docs/backend_integration.md): TurboQuant
  backend contract.
- [`docs/smoke_test.md`](docs/smoke_test.md): backend smoke-test path.

## What This Repository Is For

This repository provides code, cleaned result tables, configs, schemas, and
analysis scripts for reproducing the T-BGMP paper tables and figures from
paper-ready CSV files. It is intended to support reproducibility and data
verification.

## What This Repository Is Not

This repository is not a new quantizer kernel implementation, not a production
inference engine, not a universal KV-cache compression library, and not a broad
LongBench/RULER benchmark method.

## Main Idea

FP16-pass/aggressive-fail sensitive cases are identified first. T-BGMP ranks
key layers by empirical risk and evaluates Top-k key protection as a diagnostic
recovery protocol. Random-k and Bottom-k use the same protection budget to test
whether the empirical ranking matters.

Sensitive cases are selected from retrieval `found` fields:

```text
FP16 found == True and aggressive uniform found == False
```

Execution status, OOM, invalid FP16 baselines, and incompatible cache
interfaces are handled separately.

## Evidence Groups

Main evidence:

- Qwen3-4B
- Qwen2.5-3B
- Qwen2.5-14B
- Llama3.2-3B

Supporting / boundary-supporting:

- Mistral
- Yi
- Zephyr
- SmolLM2
- Gemma2-9B

Excluded / boundary:

- Gemma-3-4B-it
- Qwen3.5
- InternLM
- GLM

The main conditional aggregate is 183/183 restored cases. This is not an
unconditional success rate over all models or prompts. Gemma2-9B is
incomplete key-only recovery boundary evidence, not a fifth main model.

## Quick Start

```bash
python -m pip install -r requirements.txt
python experiments/run_demo_small.py
python scripts/build_paper_tables.py
python scripts/build_figures.py
python scripts/audit_results.py
python scripts/validate_csv_schema.py
python scripts/check_paper_artifacts.py
```

## Model Setup and Full Reproduction

Model weights are not included. See `docs/model_setup.md` and set `MODEL_ROOT`
before rerunning full experiments.

The committed cleaned results are sufficient for rebuilding and auditing the
paper tables without downloading checkpoints. A full GPU rerun additionally
requires independently obtained model weights, a compatible CUDA/PyTorch
environment, and an external TurboQuant/KV-cache backend. Start from:

```bash
export MODEL_ROOT=/path/to/models
export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
python experiments/generate_retrieval_cases.py \
  --config configs/default_experiment.yaml \
  --model-path "${MODEL_ROOT}/Qwen3-4B-Instruct-2507" \
  --output /path/to/outputs/qwen3_cases.csv
python experiments/stage_c_profile_key_risk.py \
  --config configs/default_experiment.yaml \
  --model-key qwen3_4b \
  --model-root "${MODEL_ROOT}" \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --output /path/to/outputs/qwen3_risk_ranking.csv
python experiments/run_full_pipeline.py \
  --cases /path/to/outputs/qwen3_cases.csv \
  --model-path "${MODEL_ROOT}/Qwen3-4B-Instruct-2507" \
  --model-id Qwen3-4B-Instruct-2507 \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --risk-ranking /path/to/outputs/qwen3_risk_ranking.csv \
  --output /path/to/outputs/qwen3_full.csv \
  --backend turboquant
```

The `requirements.txt` command in Quick Start installs the CPU analysis
environment. Before full GPU execution, install one recorded profile and
verify the runtime rather than resolving the latest packages:

```bash
conda env create -f environment-gpu-xec.yml
conda activate tbgmp-gpu-xec
python scripts/check_runtime_lock.py --profile xec_gpu \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --model-root "${MODEL_ROOT}" --model-key qwen3_4b
```

Use `environment-gpu-sip.yml` with `--profile sip_gpu` for the recorded SIP
environment. Direct pip equivalents remain in `requirements-gpu-xec.txt` and
`requirements-gpu-sip.txt`. See [`VERSION_LOCK.md`](VERSION_LOCK.md) for
provenance and the model fingerprint policy.

The case generator defaults to the recorded `formal_hpc_v1` protocol. The exact
bounded domain snapshots and their SHA-256 manifest are committed under
[`data/formal_contexts/`](data/formal_contexts/); generation reproduces the
original seed-specific answers, source rotation, insertion jitter, context
budgets, and prompt wording with the selected model tokenizer. Stage C performs
a real key-cache profiling pass with
TurboQuant's `MSECompressor`, then computes the paper's MSE, inner-product, and
effective-dimension risk score. Demo contexts remain under `data/demo/` but are
not referenced by the formal default configuration. The Stage C profiling
source must itself contain at least the configured
number of tokens; the profiler does not repeat short text.

The full runner loads each model/tokenizer once per backend process and creates
a fresh KV cache for every policy/case combination. It writes every attempt to
a durable JSONL journal, periodically atomically replaces the latest-row CSV,
and resumes by skipping only rows marked `completed=True`. OOM and execution
errors remain explicit incomplete attempts and cause a non-zero final exit so
the same command can safely be resubmitted. `configs/default_experiment.yaml`
drives the discovery/aggressive/safe policy sets, grid, Top-k limit, seeds,
output paths, and checkpoint interval. Stage F is produced automatically as
`*_stage_f.csv` and `*_stage_f_summary.csv`.

Every invocation atomically updates `*.run.json` with content hashes for all
protocol inputs, Git and model identity, runtime/CUDA/GPU versions, sanitized
arguments, SLURM identifiers, timestamps, and requested/actual context-token
ranges. Resume attempts are appended to this manifest without exposing local
absolute paths or credentials.

Model repository IDs, Hugging Face and ModelScope download examples, gated
license notes, recommended directories, and environment requirements are
documented in [`docs/model_setup.md`](docs/model_setup.md). Copy
`configs/paths_template.yaml` outside version control before adding real
machine paths.

## Running Full Experiments with TurboQuant

Full model execution requires the external TurboQuant PyTorch runtime:

https://github.com/tonbistudio/turboquant-pytorch

This repository provides T-BGMP pipeline scripts, configs, adapters, raw-output
conversion, and audit tools. It does not vendor TurboQuant, model weights,
production quantizer kernels, or raw cluster logs.

The current adapter validates the external installation but does not silently
approximate T-BGMP's arbitrary key-layer protection with TurboQuant's published
first/last-layer protection option. Real generation requires the provided patch
or an equivalent backend extension; the included binding is intended for
minimal smoke validation rather than full production inference.

A small XEC backend smoke test has been performed with a patched TurboQuant
runtime on Qwen2.5-3B-Instruct. It validates FP16 generation, T-BGMP Top-k
protected key-layer generation, raw JSONL creation, and case-level CSV
conversion on a small example. This is not a full rerun of all paper-scale
experiments. See [`examples/smoke_test/`](examples/smoke_test/).

See:

- [`docs/model_setup.md`](docs/model_setup.md)
- [`docs/backend_integration.md`](docs/backend_integration.md)
- [`docs/turboquant_api_findings.md`](docs/turboquant_api_findings.md)
- [`docs/turboquant_patch_guide.md`](docs/turboquant_patch_guide.md)
- [`docs/smoke_test.md`](docs/smoke_test.md)
- [`docs/command_cookbook.md`](docs/command_cookbook.md)

Patch deployment is pinned and executable:

```bash
python scripts/manage_turboquant_patch.py \
  --turboquant-root "${TURBOQUANT_ROOT}" --apply
```

This verifies the upstream commit and patch hash, performs an idempotent
`git apply --check`/apply flow, and tests the patched API and key-only precision
behavior before the backend is marked ready.

## Reproducibility Levels

This repository supports reproducibility at three levels:

1. **Level 1 - paper tables and figures.** Rebuild paper-facing summaries from
   the canonical cleaned CSV files in `results/paper_tables/`.
2. **Level 2 - case-level audit.** Recompute the main 183/183 aggregate and the
   Gemma2 boundary numbers from sanitized case-level discovery, Top-k, control,
   and risk files. Source file hashes are recorded without exposing source
   paths.
3. **Level 3 - model rerun interface.** Use the provided configs, pipeline CLI,
   backend adapter contract, and Slurm templates with user-supplied model
   weights, GPU environment, and a compatible KV-cache quantization backend.

Level 3 is supported as a documented integration path, not as a guaranteed
one-command reproduction on every machine. Model-specific cache APIs and
quantizer kernels remain external. Current limitation: arbitrary risk-ranked
protected key-layer ID execution depends on backend support or the provided
patch guide.
The minimal demo does not load language models. Full GPU/model execution
requires external model weights and backend support.

The complete model-free Stage A--F demonstration is:

```bash
python experiments/stage_a_discovery.py
python experiments/stage_b_mine_sensitive_cases.py
python experiments/stage_c_profile_key_risk.py --demo-stats
python experiments/stage_d_topk_recovery.py
python experiments/stage_e_random_bottom_controls.py
python experiments/stage_f_efficiency_analysis.py
```

## Repository Layout

```text
src/tbgmp/              Core selection, risk, policy, control, and metric code
experiments/            Model-free Stage A--F demonstrations
configs/                Generic configs and cleaned policy examples
data/demo/              Tiny synthetic and cleaned metadata inputs
data/schema/            JSON schemas and result-schema notes
results/paper_tables/   Canonical cleaned paper-ready CSV files
results/main_evidence/  Sanitized per-case evidence for four main models
results/supporting/     Supporting summaries and Gemma2 case-level evidence
results/audit/          Small demo outputs and regenerated integrity reports
docs/                    Protocol, interpretation, limitations, and provenance
slurm/                   Validated XEC/SIP launch templates and usage notes
```

## Paper Results Map

The current camera-ready paper is mapped to the repository as follows:

| Paper item | Canonical source |
|---|---|
| Table 1: same-budget recovery | `results/paper_tables/table_control_statistics.csv` |
| Table 2: first-success budget | `results/paper_tables/table_first_success_k.csv` |
| Figure 2: domain recovery | `results/paper_tables/figure_2_domain_recovery.csv` |
| Table 3: risk ablation | `results/paper_tables/table_risk_ablation.csv` |
| Table 4: weight sensitivity | `results/paper_tables/table_weight_sensitivity.csv` |
| Table 5: domain-held-out | `results/paper_tables/table_domain_heldout.csv` |
| Table 6: frozen Top3 | `results/paper_tables/table_frozen_top3.csv` |
| Table 7: RULER-style transfer | `results/paper_tables/table_ruler_transfer.csv` |
| Fixed-Layer Check | `results/extensions/fixed_layer/case_level.csv` |
| Cross-seed held-out | `results/extensions/cross_seed_heldout/` |
| Additional model families | `results/paper_tables/table_supporting_models.csv` |
| Gemma2 incomplete key-only recovery | `results/extensions/gemma2_boundary/` |

Tables 4--7 and the two text-only validation checks have reduced case-level evidence under
`results/extensions/`. Run `python scripts/audit_extension_results.py` to
reconstruct their reported values. The recovered path-independent experiment
runners and exact protocols are documented in
`docs/extension_experiments.md`.

Run `python scripts/check_paper_artifacts.py` to verify that every Table 1--7,
Figure 2, and text-only result mapping points to an existing artifact and
analysis command.

`tables/paper/` and `figures/paper/` are generated locally by the build
scripts and are intentionally not versioned.

## Reproducibility Note

The repository includes cleaned paper-ready CSV files and scripts for
reproducing paper tables, figures, case-level aggregates, and consistency
audits. Large model
checkpoints, raw cluster logs, and full raw model outputs are not included.
Reproducing the GPU model-execution stage requires an external compatible
KV-cache quantization runtime and independently obtained model weights.

See `docs/model_setup.md` and test the runner schema with:

```bash
python experiments/run_full_pipeline.py \
  --cases data/demo/full_runner_cases.csv \
  --model-path /path/to/models/example \
  --model-id example-model \
  --output /path/to/outputs/dry_run.csv \
  --dry-run
```

## Safety Note

No model weights, private keys, tokens, personal paths, or raw cluster logs are
included. The generic Slurm files use placeholders and do not contain account
or project paths.
