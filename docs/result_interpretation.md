# Result Interpretation

## Main Evidence

The main conditional evidence contains four pre-designated models:

| Model | Sensitive cases | Restored |
| --- | ---: | ---: |
| Qwen3-4B | 72 | 72/72 |
| Qwen2.5-3B | 72 | 72/72 |
| Qwen2.5-14B | 14 | 14/14 |
| Llama3.2-3B | 25 | 25/25 |

The 183/183 aggregate applies only to FP16-pass/aggressive-fail cases in these
four models. It is not an average accuracy over all discovery cases, all
models, or general long-context tasks.

## Supporting Models

Mistral, Yi, Zephyr, and SmolLM2 provide supporting evidence with incomplete,
small, or subset-conditioned recovery/control sets. They are not included in
the 183-case aggregate.

## Boundary-Supporting Model

Gemma2-9B is a value-bottleneck boundary case:

- key-only Top1--Top12: 7/72 unique cases
- Uniform K6/V2: 7/72
- Uniform K6/V4: 72/72

This result limits the key-only claim and indicates that some failures require
higher value precision.

## Excluded Models

Gemma-3-4B-it has an invalid FP16 baseline for the evaluated task. Qwen3.5,
InternLM, and GLM are excluded because of cache or generation compatibility
boundaries. These rows are not counted as T-BGMP retrieval failures.

## Control Interpretation

Top-k, Random-k, and Bottom-k controls use matched layer-protection budgets.
Strong Top-k separation supports the ranking signal; non-trivial Random-k or
Bottom-k rates indicate weaker evidence rather than a failed experiment.
