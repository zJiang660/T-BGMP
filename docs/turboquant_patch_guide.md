# TurboQuant Patch Guide

Status: pinned, machine-checkable patch with smoke validation. The patch is
bound to upstream commit `999713889a18c0ffa20c62a65e7cbbe5746794e3` and its
SHA-256 is recorded in `configs/runtime_lock.yaml`. A minimal XEC A800-class
backend smoke test has run successfully with patched TurboQuant.

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

## Reproducible Deployment

Start from the pinned upstream state, without local tracked modifications:

```bash
git clone https://github.com/tonbistudio/turboquant-pytorch.git \
  /path/to/turboquant-pytorch
git -C /path/to/turboquant-pytorch checkout \
  999713889a18c0ffa20c62a65e7cbbe5746794e3
```

Check compatibility without modifying the checkout:

```bash
python scripts/manage_turboquant_patch.py \
  --turboquant-root /path/to/turboquant-pytorch
```

The unmodified checkout reports `ready_to_apply` and exits non-zero because it
is not yet usable for T-BGMP. Apply and validate in one operation:

```bash
python scripts/manage_turboquant_patch.py \
  --turboquant-root /path/to/turboquant-pytorch --apply
```

The command refuses a different upstream commit, a modified patch hash, an
incompatible diff, or unrelated tracked changes. Re-running it is safe: an
already applied patch is detected with `git apply --reverse --check` and is not
applied twice. By default, success additionally requires:

1. `TurboQuantV3` and `V3Cache` to expose `protected_layer_ids` and
   `protected_key_bits` in their constructor signatures;
2. a selected layer to use K8/V2 under a K4/V2 base policy;
3. an unselected layer to remain K4/V2.

Use `--skip-runtime-validation` only while preparing a source checkout before
installing its Python dependencies. It must not be used as evidence that a GPU
environment is ready.

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
- Locked upstream commit: `999713889a18c0ffa20c62a65e7cbbe5746794e3`.
- Patch integrity and apply check: enforced by
  `scripts/manage_turboquant_patch.py`.
- API signature and key-only behavior check: enforced by the patch manager and
  `TurboQuantBackend.check_available()`.
- Runtime smoke test: PASS for a minimal XEC A800-class Qwen2.5-3B-Instruct
  smoke test using FP16 and `tbgmp_topk` with protected layer IDs `[25, 2]`.
- Full paper-scale validation: NOT TESTED by this smoke test.
