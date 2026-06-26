# TurboQuant Patch Guide

Status: proposed patch guide with smoke validation. `git apply --check` passes
against a clean clone of the inspected public TurboQuant repository, and a
minimal XEC A800-class backend smoke test has run successfully with patched
TurboQuant.

T-BGMP needs arbitrary risk-ranked key-layer protection. The inspected public
TurboQuant runtime exposes prefix/suffix protected layer counts, so exact
T-BGMP execution requires a small backend extension.

## Required Behavior

The backend should support these policy fields:

```yaml
default_key_bits: 4
default_value_bits: 2
protected_key_bits: 8
protected_layer_ids: [0, 18, 35, 7]
residual_window: 128
```

For each cache update:

1. If `layer_idx` is in `protected_layer_ids`, quantize keys with
   `protected_key_bits`.
2. If `layer_idx` is not in `protected_layer_ids`, quantize keys with
   `default_key_bits`.
3. Quantize values with `default_value_bits` for all layers unless a separate
   value policy explicitly overrides it.
4. Preserve residual-window behavior.
5. Report the effective policy in raw JSONL output.

## Implementation Checklist

1. Locate the KV-cache quantization function.
2. Replace prefix/suffix `protected_layers` logic with explicit
   `protected_layer_ids` membership.
3. Apply `protected_key_bits` only to selected key layers.
4. Keep values at default value precision unless explicitly configured.
5. Preserve `residual_window` behavior.
6. Validate with `experiments/smoke_test_backend.py`.
7. Convert raw JSONL with `scripts/convert_raw_outputs_to_case_csv.py`.
8. Audit case-level CSV with `scripts/audit_results.py`.

## Minimal Validation

Run a single prompt under FP16 first. Then run one aggressive policy and one
Top-k policy with explicit `protected_layer_ids`. Confirm:

- the raw JSONL contains the requested policy fields;
- the response is a real generated response, not a placeholder;
- `found` can be recomputed from `answer` and `response`;
- values remain at the default value precision for protected key layers;
- changing protected IDs changes the effective cache policy.

Do not claim a full GPU reproduction until this smoke test passes on the target
machine.

## Patch Check Status

- Patch file: `patches/turboquant_arbitrary_protected_layers.patch`
- Apply check: PASS against the inspected public TurboQuant repository.
- Runtime smoke test: PASS for a minimal XEC A800-class Qwen2.5-3B-Instruct
  smoke test using FP16 and `tbgmp_topk` with protected layer IDs `[25, 2]`.
- Full paper-scale validation: NOT TESTED by this smoke test.
