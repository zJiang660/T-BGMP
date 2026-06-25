# Model Download and GPU Environment

Model weights are not included in this repository. Full GPU experiments require
users to obtain the checkpoints independently, accept the applicable licenses,
and configure their locations locally. Never commit downloaded checkpoints,
access credentials, or machine-specific paths.

## Configure `MODEL_ROOT`

Keep models outside the repository and expose their parent directory through
`MODEL_ROOT`. A recommended layout is:

```text
/path/to/T-BGMP/
/path/to/models/
  Qwen3-4B-Instruct-2507/
  Qwen2.5-3B-Instruct/
  Qwen2.5-14B-Instruct/
  Llama-3.2-3B-Instruct/
  Mistral-7B-Instruct-v0.3/
  Yi-1.5-9B-Chat-16K/
  zephyr-7b-beta/
  SmolLM2-1.7B-Instruct/
  gemma-2-9b-it/
  glm-4-9b-chat-hf/
/path/to/outputs/
```

Set the variable before a full rerun:

```bash
export MODEL_ROOT=/path/to/models
```

```powershell
$env:MODEL_ROOT = "/path/to/models"
```

Use `configs/paths_template.yaml` as a portable template. Keep any file
containing real machine paths outside version control.

## Models Used by the Study

The repository ID is the canonical Hugging Face identifier. ModelScope can be
used as an alternative source when the same official checkpoint or a verified
mirror is available. Do not silently substitute a quantized, converted, or
different instruction-tuned variant: record the source, revision, and local
folder name with each rerun.

| Evidence role | Model | Canonical repository ID | Download source | Purpose and notes |
|---|---|---|---|---|
| Main | Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | Hugging Face or official Qwen ModelScope card | Main risk-ranking, recovery, and matched-budget control evidence. Use the exact `2507` instruction checkpoint. |
| Main | Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face or official Qwen ModelScope card | Long-context sensitive recovery evidence at the smaller Qwen2.5 scale. |
| Main | Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | Hugging Face or official Qwen ModelScope card | Larger-model recovery evidence. Plan substantially more GPU memory than for 3B/4B models. |
| Main | Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Hugging Face; use a verified ModelScope mirror only if its license and files match | Cross-family recovery evidence. The Hugging Face checkpoint is access-controlled and requires acceptance of Meta's license. |
| Supporting | Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Hugging Face or a verified ModelScope mirror | Supporting/tolerant-model analysis. Preserve the `v0.3` tokenizer and checkpoint revision. |
| Supporting | Yi-1.5-9B-Chat-16K | `01-ai/Yi-1.5-9B-Chat-16K` | Hugging Face or a verified ModelScope mirror | Supporting long-context model. Use the 16K chat checkpoint, not the base or shorter-context variant. |
| Supporting | Zephyr-7B-beta | `HuggingFaceH4/zephyr-7b-beta` | Hugging Face or a verified ModelScope mirror | Tolerant-model/boundary analysis. Keep the beta instruction checkpoint used by the study. |
| Supporting | SmolLM2-1.7B-Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Hugging Face or a verified ModelScope mirror | Small-model supporting evidence and low-memory compatibility checks. |
| Boundary | Gemma2-9B | `google/gemma-2-9b-it` | Hugging Face; use a verified ModelScope mirror only if terms and files match | Value-bottleneck boundary evidence, not a main recovery model. Access requires acceptance of Google's Gemma terms. |
| Boundary | GLM-4-9B-Chat-HF | `zai-org/glm-4-9b-chat-hf` | Hugging Face or a verified ModelScope mirror | Boundary/compatibility analysis. The former `THUDM/glm-4-9b-chat-hf` identifier resolves to the current repository. Requires `transformers>=4.46.0` according to its model card. |

Other excluded or boundary models mentioned in the paper should be downloaded
only when reproducing those specific compatibility checks. They are not needed
to rebuild the paper tables from the committed cleaned results.

## Download from Hugging Face

Install the official client, complete any required license acceptance in the
browser, and download into `MODEL_ROOT`:

```bash
python -m pip install --upgrade huggingface_hub
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
```

Replace the repository ID and destination directory for each model in the
table. For gated checkpoints such as Llama and Gemma, authenticate through the
official client without placing credentials in scripts, YAML files, logs, or
Git history.

## Download from ModelScope

ModelScope is useful when it is the more reliable source in the execution
region. Install its official client and use the model card's exact identifier:

```bash
python -m pip install --upgrade modelscope
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 \
  --local_dir "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
```

Qwen publishes official ModelScope cards using the Qwen namespace. For other
families, confirm that a ModelScope entry is official or a faithful mirror
before use. Compare `config.json`, tokenizer files, weight format, parameter
count, license, and revision against the canonical model card.

## Software Environment

The model-free analysis environment is defined by `requirements.txt` and
`environment.yml`. Full inference additionally requires:

- a CUDA-capable NVIDIA driver and a matching CUDA-enabled PyTorch build;
- `transformers`, `accelerate`, and `safetensors` versions compatible with the
  selected checkpoint;
- tokenizer dependencies required by the selected family;
- `bitsandbytes` only when the chosen loading path uses 4-bit/8-bit weights;
- the external TurboQuant/KV-cache backend used for execution;
- model-specific remote code only when required by the official model card.

The exact PyTorch, CUDA, Transformers, and backend versions depend on the GPU
driver and cache implementation. Record the complete environment, model
revision, quantization policy, requested context, actual context, and random
seed for every rerun. Follow the individual model card when it imposes a newer
minimum Transformers version.

## Hardware Planning

- The small demo, table builders, and audits run on CPU without model weights.
- Full 1.7B--4B experiments require a CUDA GPU with enough memory for model
  weights, activations, the requested context, and both FP16 and compressed
  cache modes.
- 7B--14B models and long contexts require substantially more memory. Use the
  same context, loading precision, backend, and fallback policy when making
  cross-model comparisons.
- An out-of-memory attempt is an execution outcome, not a retrieval miss. Log
  it separately and retain the actual context length of any fallback attempt.

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

The repository does not implement a universal quantizer kernel. The adapter
must apply `QuantizationConfig` to a compatible model/cache runtime. An
implementation template is provided in `docs/backend_adapter_template.py`.

## Dry Run Before GPU Execution

Test the CLI, path handling, and output schema without loading a model:

```bash
python experiments/run_full_pipeline.py \
  --cases data/demo/full_runner_cases.csv \
  --model-path "${MODEL_ROOT}/example-model" \
  --model-id example-model \
  --output /path/to/outputs/stage_a.csv \
  --dry-run
```
