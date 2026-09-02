# Demo Data

This directory contains small, cleaned metadata files used to inspect the
case-selection and evidence-packaging workflow without loading model weights.

- `demo_cases.csv` is the canonical small input for the model-free Stage A--F
  scripts.
- `demo_expected_outputs.csv` records the expected sensitive-case and recovery
  classifications.
- `demo_context_math.txt` is a tiny synthetic retrieval context.
- `contexts/` contains four small public synthetic sources for exercising the
  tokenizer-aware domain/context/depth/seed case generator.

These files do not contain model weights, model caches, raw cluster logs,
credentials, or private machine paths.
