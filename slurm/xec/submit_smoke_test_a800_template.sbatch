#!/usr/bin/env bash
# Sanitized template for a minimal backend smoke test.
# Adapt partition/account, environment setup, and paths for the target cluster.
#SBATCH --job-name=tbgmp_smoke
#SBATCH --partition=<your_partition>
#SBATCH --account=<your_account>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/tbgmp_smoke_%j.out
#SBATCH --error=logs/tbgmp_smoke_%j.err

set -euo pipefail

export TBGMP_ROOT=/path/to/T-BGMP
export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${TBGMP_ROOT}"
mkdir -p "${OUTPUT_ROOT}/smoke_test" logs

"${PYTHON_BIN}" experiments/smoke_test_backend.py \
  --backend turboquant \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --model-key qwen25_3b \
  --prompt "The hidden answer is ABC123. What is the hidden answer?" \
  --answer ABC123 \
  --policy fp16 \
  --max-new-tokens 12 \
  --output "${OUTPUT_ROOT}/smoke_test/fp16_raw.jsonl"

"${PYTHON_BIN}" experiments/smoke_test_backend.py \
  --backend turboquant \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --model-key qwen25_3b \
  --prompt "The hidden answer is ABC123. What is the hidden answer?" \
  --answer ABC123 \
  --policy tbgmp_topk \
  --protected-layer-ids 25,2 \
  --default-key-bits 4 \
  --default-value-bits 2 \
  --protected-key-bits 6 \
  --residual-window 128 \
  --max-new-tokens 12 \
  --output "${OUTPUT_ROOT}/smoke_test/tbgmp_topk_raw.jsonl"

"${PYTHON_BIN}" scripts/convert_raw_outputs_to_case_csv.py \
  --input "${OUTPUT_ROOT}/smoke_test/fp16_raw.jsonl" \
  --output "${OUTPUT_ROOT}/smoke_test/fp16_case_level.csv"

"${PYTHON_BIN}" scripts/convert_raw_outputs_to_case_csv.py \
  --input "${OUTPUT_ROOT}/smoke_test/tbgmp_topk_raw.jsonl" \
  --output "${OUTPUT_ROOT}/smoke_test/tbgmp_topk_case_level.csv"
