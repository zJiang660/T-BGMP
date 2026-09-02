# Experiment Protocol

## Scope

T-BGMP is a diagnostic failure-recovery protocol for cases where FP16 retrieval
succeeds and aggressive uniform KV-cache quantization fails. It is not a new
quantizer kernel, a speedup implementation, or an oracle-free deployment
policy.

## Stage A: Discovery

Evaluate FP16 and predefined uniform precision policies over a controlled grid
of domains, context lengths, needle depths, and seeds. Record retrieval
`found`, execution validity, errors, and memory statistics separately.
The model-free implementation in this repository validates answer matching on
the synthetic demo rows; full GPU discovery requires an external runner.
`experiments/run_full_pipeline.py` executes FP16, K2/V2, K4/V2, K6/V2, and
K6/V4 through the user-supplied backend.

Generate the grid with `experiments/generate_retrieval_cases.py`. It uses the
selected model tokenizer, deterministic hidden strings, configured domain text
sources, and exact depth/seed combinations. During execution, the runner
renders each request with `configs/prompt_template.yaml` and the tokenizer's
chat template, with thinking disabled where the model exposes that option.
Generated case files record `document_tokens`; real backend outputs separately
record `actual_context_tokens` after the complete rendered prompt is tokenized.
The runner marks rendered prompts so the backend does not add special tokens a
second time.

## Stage B: Sensitive-Case Mining

Sensitive cases are selected using `found` fields, not execution status:

```text
FP16 found == True and aggressive uniform found == False
```

An OOM, invalid FP16 baseline, or incompatible cache interface is not a
retrieval miss and is not admitted to the conditional recovery set.

## Stage C: Key-Risk Profiling

For each key layer, estimate upper-tail reconstruction MSE, inner-product
distortion, and effective dimension where available. Normalize these components
within a model and sum them to obtain a model-specific empirical risk score.
`experiments/stage_c_profile_key_risk.py` can either consume precomputed layer
statistics or run the complete model profiling path. The real path extracts key
tensors from `past_key_values`, reconstructs them with TurboQuant's
`MSECompressor`, applies the bit-normalized `2^(2b)` factor, and reports the 95th
percentile MSE and inner-product errors plus participation-ratio effective
dimension. The default formal profiling context is literature at 4096 tokens.
The profiling source must contain at least the requested number of tokens; it
is truncated but never repeated, because repetition would change key statistics.
The bundled four-layer statistics are available only through the explicit
`--demo-stats` flag and are never selected silently by the formal path.

## Stage D: Top-k Recovery

Sort key layers by decreasing risk. Starting from the aggressive default
precision, raise only the Top-k key layers to the protected precision and
measure the first successful recovery budget.
The full runner evaluates Top1 through Top12 (or the configured maximum)
without stopping after the first success, preserving the diagnostic sweep.
The runner preserves the selected case's aggressive K/V bits: K2/V2 failures
remain K2/V2 outside protected key layers, while K4/V2 failures remain K4/V2.

T-BGMP is a diagnostic recovery protocol, not an oracle-free deployment policy:
the current procedure evaluates recoverability after a failure is known.

## Stage E: Random/Bottom Controls

At the same protection budget, compare risk-ranked Top-k layers with random
layers and bottom-ranked layers. These controls test whether ranking matters,
not merely whether protecting any layers helps.
The full runner evaluates three seeded Random-k policies and one Bottom-k
policy at the first-success Top-k budget.

## Stage F: Efficiency Analysis

For successfully recovered cases, compare T-BGMP KV-cache saving with a
uniform safe precision such as K6/V2. Runtime is diagnostic only; this
repository makes no system speedup claim.
The full runner pairs each first-success T-BGMP policy with the configured
K6/V2 and K6/V4 rows for the same case. A comparison is valid only when both
retrievals completed successfully. Backend-reported KV saving is preferred;
otherwise the output is explicitly labeled as nominal bit-budget accounting.

## Runtime and Resume Semantics

The production backend owns one lazily initialized model/tokenizer runtime per
model path. Policies rebuild or reset only the KV cache. Each attempted
combination is fsynced to a JSONL journal immediately; the CSV is an atomic
latest-attempt snapshot. On restart, only `completed=True` identities are
skipped. OOM and other errors are preserved with `completed=False` and retried
on the next invocation. Remaining incomplete rows make the runner exit nonzero
after all durable outputs and metadata have been written.

Each run also creates an atomic `*.run.json` provenance manifest. The manifest
records SHA-256 fingerprints for the experiment, case, policy, prompt, backend,
model-registry, and ranking inputs; the repository commit and dirty state; model
configuration identity; Python, package, CUDA, and GPU versions; sanitized
command arguments; SLURM job identifiers; invocation timestamps; and requested
versus actually observed context-token ranges. Resuming the same output appends
a new invocation record. Protocol-changing inputs retain signature protection,
while machine-specific absolute paths and credentials are not stored.

## Evidence Groups

Main evidence models:

- Qwen3-4B
- Qwen2.5-3B
- Qwen2.5-14B
- Llama3.2-3B

Supporting models:

- Mistral
- Yi
- Zephyr
- SmolLM2

Boundary and excluded models:

- Gemma2-9B is boundary-supporting value-bottleneck evidence, not main
  evidence.
- Gemma-3-4B-it has an invalid FP16 baseline for the evaluated task.
- Qwen3.5, InternLM, and GLM are excluded because of cache or generation
  interface limitations.

## Known Limitations

- The evaluated task is exact needle-style retrieval.
- Risk rankings are model-specific.
- Calibration and evaluation distributions are not fully separated.
- Random controls pool available seeds in some summaries.
- Key-only protection does not repair every quantization failure mode.
