# T-BGMP

T-BGMP is a diagnostic failure-recovery protocol for aggressive KV-cache
quantization.

## What This Repository Contains

- Core analysis code for empirical layer risk scoring
- Correct FP16-pass/aggressive-fail sensitive-case selection
- Top-k, random, and bottom-layer policy utilities
- Cleaned paper-ready results
- Scripts for regenerating paper summaries and a control figure
- A small demo that does not load a language model

## What This Repository Does Not Contain

- Model weights or model caches
- Raw cluster logs
- Private paths, credentials, or access keys
- Full model responses or large experiment outputs
- A quantizer kernel or inference speedup implementation

## Main Idea

FP16-pass/aggressive-fail sensitive cases are identified first. T-BGMP ranks
key layers by empirical risk and evaluates Top-k key protection as a diagnostic
recovery protocol. Values and unprotected keys remain at the aggressive default
precision.

The repository does not claim that T-BGMP is a universal KV-cache quantization
library, an oracle-free deployment policy, or a broad long-context benchmark
improvement method.

## Main Evidence

The cleaned tables include conditional recovery results for:

- Qwen3-4B: 72/72
- Qwen2.5-3B: 72/72
- Qwen2.5-14B: 14/14
- Llama3.2-3B: 25/25

## Supporting and Boundary Cases

Supporting analyses cover Mistral, Yi, Zephyr, and SmolLM2. Gemma2-9B is
reported as boundary-supporting value-bottleneck evidence. Invalid or
interface-limited executions for Gemma-3-4B-it, Qwen3.5, InternLM, and GLM are
not counted as retrieval failures.

## Quick Start

```bash
python -m pip install -r requirements.txt
python experiments/run_demo_small.py
python scripts/build_paper_tables.py
python scripts/build_figures.py
```

## Repository Layout

```text
src/tbgmp/             Core analysis utilities
experiments/           Small model-free demonstration
scripts/               Paper summary and figure regeneration
results/paper_tables/  Cleaned paper-ready CSV files
figures/paper/          Paper figures and reproduced control plot
docs/                   Experiment protocol and scope
```

## Reproducibility Note

This minimal repository reproduces the analysis and paper-table generation
pipeline from cleaned CSV files. It does not include large model checkpoints or
raw cluster outputs. Reproducing the model execution stage requires a separate
compatible KV-cache quantization environment and locally obtained model
weights.
