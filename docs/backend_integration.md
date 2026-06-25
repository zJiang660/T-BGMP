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

The included adapter therefore validates `TURBOQUANT_ROOT` and imports the
external runtime, but deliberately raises `NotImplementedError` at generation
until that exact cache-policy binding is implemented locally. It does not
approximate Top-k with first/last layers and does not fabricate output.

The minimal repository provides adapters and pipeline scripts, but full GPU
execution depends on the external backend and user-supplied model weights.
