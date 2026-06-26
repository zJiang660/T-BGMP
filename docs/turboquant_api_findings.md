# TurboQuant API Findings

This note records the public TurboQuant API boundary used by the T-BGMP
integration path.

## Repository Checked

- TurboQuant repository: <https://github.com/tonbistudio/turboquant-pytorch>
- Files inspected:
  - `turboquant/compressors_v3.py`
  - `turboquant/generation_test.py`
  - `turboquant/generation_test_v2.py`
  - `turboquant/validate_v3.py`

## Findings

| Capability | Status | Evidence / interpretation |
|---|---:|---|
| `TurboQuantV3` compressor | yes | `compressors_v3.py` exposes `class TurboQuantV3`. |
| K/V bit split | yes | The constructor accepts `key_bits` and `value_bits`. |
| K2/V2, K4/V2, K6/V2, K6/V4 style settings | partial | The API accepts separate key/value bit widths. Example configs include K4/V2, K6/V4, K8/V4, and related settings. |
| `residual_window` | yes | The compressor and cache examples expose `residual_window`. |
| `protected_layers` | prefix/suffix count | The inspected implementation treats `protected_layers` as a number of first and last layers to protect. |
| Arbitrary protected layer IDs | no direct public binding found | No inspected public entry point accepts a ranked list such as `[0, 24, 1, 36]`. |
| Key-only protected precision | no direct public binding found | The inspected protection path raises both key and value precision for protected prefix/suffix layers. |
| Keep values at V2 while protecting selected keys | not directly supported by inspected public API | T-BGMP requires selected key layers to use protected key precision while values remain at the default value precision. |

## Protected Layer Semantics

The public `protected_layers` argument does not directly implement
risk-ranked arbitrary key-layer ID protection. T-BGMP therefore requires either
a small adapter/patch or a backend extension to apply arbitrary protected key
layer IDs.

The required semantics are:

1. Accept `protected_layer_ids` as explicit layer IDs.
2. Apply `protected_key_bits` only to those key layers.
3. Keep values at `default_value_bits` unless a separate value policy is
   explicitly requested.
4. Preserve `residual_window` behavior.
5. Return raw model outputs that can be converted into sanitized case-level CSV.

## Required Patch

Required patch: yes, for exact T-BGMP execution with arbitrary risk-ranked
key-layer IDs.

The public repository is still usable as the external runtime basis, but the
current T-BGMP adapter refuses to approximate Top-k protection with
first/last-layer protection. See `docs/turboquant_patch_guide.md`.

Patch apply check: PASS against a clean clone of the inspected public
TurboQuant repository.

Patch runtime validation: PASS for a minimal XEC A800-class backend smoke test
with Qwen2.5-3B-Instruct, FP16, and `tbgmp_topk` protected key layers `[25, 2]`.
This validates the small integration path only; it is not a full paper-scale
rerun.
