# Raw Output Schema

Full-model runners should write one JSON object per line. The schema is defined
in `data/schema/raw_output_schema.json`.

```json
{
  "model": "qwen25_3b",
  "case_id": "math_ctx4096_depth50_seed0",
  "domain": "math",
  "context_length": 4096,
  "actual_context_length": 4096,
  "depth": 50,
  "seed": 0,
  "policy": "uniform_k4_v2_rw128",
  "policy_type": "uniform",
  "key_bits": 4,
  "value_bits": 2,
  "protected_key_bits": null,
  "protected_layers": [],
  "residual_window": 128,
  "answer": "needle_answer",
  "response": "model output text",
  "status": "success",
  "metadata": {}
}
```

The raw JSONL may contain complete responses and should remain outside version
control. Convert it into a compact case-level CSV with:

```bash
python scripts/convert_raw_outputs_to_case_csv.py \
  --input /path/to/outputs/raw/stage_a_qwen25_3b.jsonl \
  --output /path/to/outputs/case_csv/stage_a_discovery.csv
```

The converter recomputes `found` from `answer` and `response`, omits the full
response and metadata, and retains only a short `response_excerpt`. It does not
copy execution paths, environment values, or authentication material.
