# T-BGMP

T-BGMP is a diagnostic failure-recovery protocol for aggressive KV-cache
quantization.

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
value-bottleneck boundary evidence, not a fifth main model.

## Quick Start

```bash
python -m pip install -r requirements.txt
python experiments/run_demo_small.py
python scripts/build_paper_tables.py
python scripts/build_figures.py
python scripts/audit_results.py
python scripts/validate_csv_schema.py
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
python experiments/run_full_pipeline.py \
  --cases data/demo/full_runner_cases.csv \
  --model-path "${MODEL_ROOT}/Qwen3-4B-Instruct-2507" \
  --model-id Qwen3-4B-Instruct-2507 \
  --output /path/to/outputs/qwen3_full.csv \
  --backend module.path:factory
```

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
python experiments/stage_c_profile_key_risk.py
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
results/boundary_excluded/ Compatibility and invalid-boundary summaries
results/audit/          Generated audit summary and artifact hashes
figures/paper/           Paper figures and reproduced control plot
tables/paper/            CSV copies and generated Markdown tables
docs/                    Protocol, interpretation, limitations, and provenance
slurm/xec/               Sanitized templates only
```

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
