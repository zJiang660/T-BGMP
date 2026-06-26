# Command Cookbook

## A. Level 1/2 Lightweight Reproduction

This path rebuilds the public paper artifacts and audits the sanitized
case-level evidence. It does not load model weights.

```bash
python -m pip install -r requirements.txt
python experiments/run_demo_small.py
python scripts/audit_results.py
python scripts/build_paper_tables.py
python scripts/build_figures.py
python scripts/validate_csv_schema.py
```

## B. Prepare TurboQuant

Clone the external runtime and configure paths outside version control:

```bash
git clone https://github.com/tonbistudio/turboquant-pytorch.git \
  /path/to/turboquant-pytorch

export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
```

PowerShell:

```powershell
$env:TURBOQUANT_ROOT="/path/to/turboquant-pytorch"
$env:MODEL_ROOT="/path/to/models"
$env:OUTPUT_ROOT="/path/to/outputs"
```

Check the current public API boundary:

```bash
python experiments/smoke_test_backend.py --help
```

## C. Full Pipeline Dry Run

Dry-run mode validates the pipeline contract and writes schema-test rows. It
does not perform model execution.

```bash
python experiments/run_full_pipeline.py \
  --dry-run \
  --backend turboquant \
  --model-key qwen25_3b \
  --model-root "$MODEL_ROOT" \
  --turboquant-root "$TURBOQUANT_ROOT" \
  --output-dir "$OUTPUT_ROOT/qwen25_3b"
```

Expected dry-run messages include:

```text
Stage A: discovery
Stage B: mine sensitive cases
Stage C: profile key risk
Stage D: Top-k recovery
Stage E: Random/Bottom controls
Stage F: efficiency analysis
Backend: turboquant
Model key: qwen25_3b
No model execution performed in dry-run mode.
```

The individual stage scripts expose compatible command-line flags:

```bash
python experiments/stage_a_discovery.py --help
python experiments/stage_b_mine_sensitive_cases.py --help
python experiments/stage_c_profile_key_risk.py --help
python experiments/stage_d_topk_recovery.py --help
python experiments/stage_e_random_bottom_controls.py --help
python experiments/stage_f_efficiency_analysis.py --help
```

## D. Real Execution Template

Requires working TurboQuant backend binding and model weights.

```bash
python experiments/smoke_test_backend.py \
  --backend turboquant \
  --turboquant-root "$TURBOQUANT_ROOT" \
  --model-root "$MODEL_ROOT" \
  --model-key qwen25_3b \
  --prompt "The hidden answer is ABC123. What is the hidden answer?" \
  --answer ABC123 \
  --policy fp16 \
  --output "$OUTPUT_ROOT/qwen25_3b/smoke_test.jsonl"
```

Convert raw JSONL to a compact case-level CSV:

```bash
python scripts/convert_raw_outputs_to_case_csv.py \
  --input "$OUTPUT_ROOT/qwen25_3b/smoke_test.jsonl" \
  --output "$OUTPUT_ROOT/qwen25_3b/smoke_test_case.csv"
```

Run the orchestrated pipeline after a compatible backend can execute arbitrary
risk-ranked key-layer protection:

```bash
python experiments/run_full_pipeline.py \
  --config configs/default_experiment.yaml \
  --cases /path/to/cases.csv \
  --model-key qwen25_3b \
  --model-root "$MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT/qwen25_3b" \
  --backend turboquant \
  --turboquant-root "$TURBOQUANT_ROOT" \
  --risk-ranking /path/to/risk_ranking.csv \
  --max-new-tokens 32 \
  --seed 0
```

The included adapter intentionally stops before generation until the exact
arbitrary key-layer protection semantics are implemented. See
`docs/turboquant_api_findings.md` and `docs/turboquant_patch_guide.md`.

For an A800-style batch template, see:

```bash
slurm/xec/submit_smoke_test_a800_template.sbatch
```

The repository includes a sanitized example from a completed small XEC smoke
test in `examples/smoke_test/`.
