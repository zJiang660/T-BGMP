# Model and GPU Setup

Model weights are not included in this repository. Users must obtain models
under their applicable licenses and provide paths at runtime.

## Recommended Directory Layout

```text
/path/to/T-BGMP
/path/to/models/
  qwen3-4b/
  qwen25-3b/
  qwen25-14b/
  llama32-3b/
/path/to/outputs/
```

Copy `configs/paths_template.yaml` outside version control if machine-specific
paths are needed.

## Software

The analysis environment is defined by `requirements.txt` and
`environment.yml`. Full inference additionally requires:

- a CUDA-compatible PyTorch build;
- a Transformers version compatible with the selected model;
- the external TurboQuant/KV-cache backend used for execution;
- optional model-specific remote code where required.

Exact torch, CUDA, and Transformers versions depend on the external backend and
GPU driver. Record them with every rerun rather than assuming the lightweight
analysis environment is sufficient.

## Hardware

- The small demo and data audits run on CPU and require no model weights.
- Full 3B/4B experiments generally require a CUDA GPU with enough memory for
  the model, context, and cache implementation.
- Larger models or long contexts may require an A800-class GPU or equivalent.

## External Backend Contract

`experiments/run_full_pipeline.py` accepts `--backend module.path:factory`.
The factory must return an object implementing:

```python
generate(
    model_path,
    prompt,
    answer,
    policy_name,
    quantization,
    max_new_tokens,
)
```

and return `tbgmp.kv_cache_wrapper.GenerationResult`.

The repository deliberately does not pretend to implement a universal
quantizer kernel. The adapter must apply `QuantizationConfig` to the user's
compatible model/cache runtime.

An implementation template is provided in
`docs/backend_adapter_template.py`.

## Dry Run

The CLI and output schema can be tested without a model:

```bash
python experiments/run_full_pipeline.py \
  --cases data/demo/full_runner_cases.csv \
  --model-path /path/to/models/example \
  --model-id example-model \
  --output /path/to/outputs/stage_a.csv \
  --dry-run
```
