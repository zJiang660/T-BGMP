# Qwen2.5 8192 top4 vs random4/bottom4 control

## Scope
- model: qwen25 / Qwen2.5-3B-Instruct
- context: 8192
- domains: math, literature, science
- answer: AURORA-7749
- top4 rows are reused from adaptive_validation_results/qwen25_8192_topk_search.csv when available; the control runner does not rerun top-k rows.

## Policy summary
- qwen25_tbgmp_len8192_top4_keys: success=3/3, mean saving=77.856, statuses=success:3
- qwen25_tbgmp_len8192_random4_keys_seed0: success=0/3, mean saving=77.856, statuses=failed:3
- qwen25_tbgmp_len8192_random4_keys_seed1: success=0/3, mean saving=77.856, statuses=failed:3
- qwen25_tbgmp_len8192_random4_keys_seed2: success=0/3, mean saving=77.856, statuses=failed:3
- qwen25_tbgmp_len8192_bottom4_keys: success=0/3, mean saving=77.856, statuses=failed:3

## Control comparison
- top4 success rate: 1.000, mean KV saving: 77.856
- best random4 success rate: 0.000 (qwen25_tbgmp_len8192_random4_keys_seed0), mean KV saving: 77.856
- bottom4 success rate: 0.000, mean KV saving: 77.856

## Missing rows
- None.
