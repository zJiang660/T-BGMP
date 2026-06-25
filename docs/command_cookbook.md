# Command Cookbook

## A. Lightweight Demo

```bash
python -m pip install -r requirements.txt
python experiments/run_demo_small.py
python scripts/audit_results.py
```

This path does not load a language model.

## B. Rebuild and Audit Paper Artifacts

```bash
python scripts/build_paper_tables.py
python scripts/build_figures.py
python scripts/audit_results.py
python scripts/validate_csv_schema.py
```

## C. Prepare Full Model Execution

```bash
git clone https://github.com/tonbistudio/turboquant-pytorch.git \
  /path/to/workspace/turboquant-pytorch

export TURBOQUANT_ROOT=/path/to/workspace/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
```

Inspect every stage interface:

```bash
python experiments/stage_a_discovery.py --help
python experiments/stage_b_mine_sensitive_cases.py --help
python experiments/stage_c_profile_key_risk.py --help
python experiments/stage_d_topk_recovery.py --help
python experiments/stage_e_random_bottom_controls.py --help
python experiments/stage_f_efficiency_analysis.py --help
```

The standalone Stage A--F scripts provide the model-free analysis and
post-processing contracts. End-to-end model orchestration is exposed through:

```bash
python experiments/run_full_pipeline.py \
  --config configs/default_experiment.yaml \
  --cases /path/to/cases.csv \
  --model-key qwen25_3b \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/qwen25_3b" \
  --backend turboquant \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --risk-ranking /path/to/risk_ranking.csv \
  --max-new-tokens 32 \
  --seed 0
```

After a backend produces raw JSONL, convert and process it stage by stage:

```bash
python scripts/convert_raw_outputs_to_case_csv.py \
  --input "${OUTPUT_ROOT}/qwen25_3b/raw/stage_a.jsonl" \
  --output "${OUTPUT_ROOT}/qwen25_3b/case_csv/stage_a_discovery.csv"

python experiments/stage_b_mine_sensitive_cases.py \
  --input "${OUTPUT_ROOT}/qwen25_3b/case_csv/discovery_wide.csv" \
  --output "${OUTPUT_ROOT}/qwen25_3b/case_csv/sensitive_cases.csv"

python experiments/stage_c_profile_key_risk.py \
  --input "${OUTPUT_ROOT}/qwen25_3b/key_distortion_stats.csv" \
  --output "${OUTPUT_ROOT}/qwen25_3b/case_csv/risk_ranking.csv"

python experiments/stage_d_topk_recovery.py \
  --cases /path/to/demo_or_declared_outcomes.csv \
  --risk-ranking "${OUTPUT_ROOT}/qwen25_3b/case_csv/risk_ranking.csv" \
  --output "${OUTPUT_ROOT}/qwen25_3b/case_csv/topk_recovery.csv"

python experiments/stage_e_random_bottom_controls.py \
  --cases /path/to/demo_or_declared_outcomes.csv \
  --risk-ranking "${OUTPUT_ROOT}/qwen25_3b/case_csv/risk_ranking.csv" \
  --output "${OUTPUT_ROOT}/qwen25_3b/case_csv/random_bottom_controls.csv"

python experiments/stage_f_efficiency_analysis.py \
  --input "${OUTPUT_ROOT}/qwen25_3b/case_csv/topk_recovery.csv" \
  --output "${OUTPUT_ROOT}/qwen25_3b/case_csv/efficiency_summary.csv"
```

These commands require user-supplied model weights, GPU hardware, and a
working TurboQuant backend integration. The included TurboQuant adapter checks
the external installation but intentionally stops before generation until
arbitrary key-layer protection is bound to the local runtime.
