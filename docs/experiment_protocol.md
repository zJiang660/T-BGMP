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

## Stage D: Top-k Recovery

Sort key layers by decreasing risk. Starting from the aggressive default
precision, raise only the Top-k key layers to the protected precision and
measure the first successful recovery budget.

T-BGMP is a diagnostic recovery protocol, not an oracle-free deployment policy:
the current procedure evaluates recoverability after a failure is known.

## Stage E: Random/Bottom Controls

At the same protection budget, compare risk-ranked Top-k layers with random
layers and bottom-ranked layers. These controls test whether ranking matters,
not merely whether protecting any layers helps.

## Stage F: Efficiency Analysis

For successfully recovered cases, compare T-BGMP KV-cache saving with a
uniform safe precision such as K6/V2. Runtime is diagnostic only; this
repository makes no system speedup claim.

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
