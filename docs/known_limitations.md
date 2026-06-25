# Known Limitations

- **Not universal.** T-BGMP is a diagnostic recovery protocol for selected
  FP16-pass/aggressive-fail cases, not a universal compression method.
- **Not a speedup result.** The repository does not implement or benchmark a
  production quantizer kernel. Quantization overhead may dominate runtime.
- **No fully held-out calibration split.** The current evidence does not enforce
  a complete separation between risk profiling and evaluation distributions.
- **Needle-style retrieval only.** The reported task is exact retrieval, not
  summarization, multi-hop reasoning, LongBench, or RULER.
- **Model-specific rankings.** Layer risk is profiled per model and is not
  assumed to transfer across architectures.
- **Gemma2 value-cache bottleneck.** Key-only protection restores only 7/72
  unique Gemma2-9B cases, while K6/V4 restores 72/72.
- **Compatibility boundaries.** Invalid FP16 baselines and incompatible cache
  or generation interfaces cannot support recovery conclusions.
- **Model execution excluded.** This minimal repository validates the analysis
  pipeline from sanitized case-level outputs and cleaned summaries; external
  model runtimes, checkpoints, and a compatible KV-cache backend are required
  for full GPU reruns.
