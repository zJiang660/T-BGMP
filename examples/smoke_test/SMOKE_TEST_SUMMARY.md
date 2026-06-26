# XEC TurboQuant Backend Smoke Test

Model: Qwen2.5-3B-Instruct

Backend: patched TurboQuant

Policies tested: `fp16`, `tbgmp_topk`

Protected layer IDs: `[25, 2]`

Result:

- raw JSONL generated: YES
- case-level CSV conversion: YES
- found computed: YES
- FP16 smoke found: YES
- T-BGMP Top-k smoke found: YES
- exact full paper reproduction: NO, smoke test only

This validates the repository's Level 3 integration path on a small XEC
A800-class GPU smoke example. It is not a full rerun of the paper-scale
experiments.
