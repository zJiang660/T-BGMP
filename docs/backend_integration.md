# Backend Integration

T-BGMP requires a KV-cache quantization backend for full model execution. The
experiments were designed around TurboQuant-style KV precision settings. Users
can install the external TurboQuant PyTorch repository:

https://github.com/tonbistudio/turboquant-pytorch

This repository does not include the TurboQuant runtime, model weights, or
production quantizer kernels. T-BGMP uses TurboQuant as an external
runtime/backend and does not vendor or maintain TurboQuant itself.

## Recommended Layout

```text
workspace/
  T-BGMP/
  turboquant-pytorch/
  models/
  outputs/
```

Clone the external dependency next to this repository:

```bash
git clone https://github.com/tonbistudio/turboquant-pytorch.git \
  /path/to/workspace/turboquant-pytorch
```

No Git submodule is used. This keeps upstream history, releases, and licensing
independent.

## Environment

```bash
export TBGMP_ROOT=/path/to/T-BGMP
export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
```

PowerShell:

```powershell
$env:TBGMP_ROOT = "/path/to/T-BGMP"
$env:TURBOQUANT_ROOT = "/path/to/turboquant-pytorch"
$env:MODEL_ROOT = "/path/to/models"
$env:OUTPUT_ROOT = "/path/to/outputs"
```

Install the CUDA/PyTorch/Transformers environment required by the selected
model and the external runtime. Follow the upstream TurboQuant repository for
its current dependencies.

## Minimum Backend Responsibilities

A complete backend must:

1. Load a user-supplied Hugging Face model and tokenizer.
2. Run FP16 generation.
3. Run uniform KV-cache policies K2/V2, K4/V2, K6/V2, and K6/V4.
4. Run protected-key Top-k policies with default key bits, default value bits,
   protected key bits, arbitrary protected layer IDs, and a residual window.
5. Return the response, execution status, retrieval result, and optionally
   runtime, throughput, peak GPU memory, and KV saving.

The request/result protocol is in `src/tbgmp/backends/base.py`. The validation
adapter is in `src/tbgmp/backends/turboquant_backend.py`.

## Current Adapter Status

The external repository currently exposes `TurboQuantV3` and example
Transformers cache wrappers. Its published `protected_layers` option represents
a count of first/last protected layers and applies one compressed cache path to
keys and values. T-BGMP requires arbitrary empirically ranked key-layer IDs
while values remain at the default precision.

Relevant upstream reference files:

- [`turboquant/compressors_v3.py`](https://github.com/tonbistudio/turboquant-pytorch/blob/master/turboquant/compressors_v3.py)
- [`turboquant/generation_test.py`](https://github.com/tonbistudio/turboquant-pytorch/blob/master/turboquant/generation_test.py)

The included adapter validates `TURBOQUANT_ROOT`, imports the external runtime,
detects whether the arbitrary key-layer patch is present, and can run a minimal
smoke generation path when the patch and required Python packages are
available. It does not approximate Top-k with first/last layers and does not
fabricate output.

The minimal repository provides adapters and pipeline scripts, but full GPU
execution depends on the external backend and user-supplied model weights.

Detailed API findings are recorded in `docs/turboquant_api_findings.md`. A
proposed extension path is documented in `docs/turboquant_patch_guide.md` and
`patches/turboquant_arbitrary_protected_layers.patch`.

## Backend Smoke Test

Use `experiments/smoke_test_backend.py` to check a local runtime:

```bash
python experiments/smoke_test_backend.py \
  --backend turboquant \
  --turboquant-root "$TURBOQUANT_ROOT" \
  --model-root "$MODEL_ROOT" \
  --model-key qwen25_3b \
  --prompt "The hidden answer is ABC123. What is the hidden answer?" \
  --answer ABC123 \
  --policy fp16 \
  --output "$OUTPUT_ROOT/smoke_test.jsonl"
```

If the backend is not bound to real generation, the smoke test fails before
writing output. If it succeeds, convert raw JSONL with
`scripts/convert_raw_outputs_to_case_csv.py`.

An XEC A800-class smoke test has been completed with Qwen2.5-3B-Instruct,
patched TurboQuant, FP16, and `tbgmp_topk` with protected layer IDs `[25, 2]`.
The sanitized example is in `examples/smoke_test/`.
