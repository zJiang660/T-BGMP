#!/usr/bin/env bash
# XEC A800 template for the canonical resumable pipeline.
# Export the required paths before submission; no private path is embedded here.
# Model weights and the TurboQuant runtime remain external to this repository.
#SBATCH --job-name=tbgmp-full
#SBATCH --partition=<your_partition>
#SBATCH --account=<your_account>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=tbgmp_pipeline_%j.out
#SBATCH --error=tbgmp_pipeline_%j.err

set -euo pipefail

: "${TBGMP_ROOT:?Export TBGMP_ROOT before sbatch}"
: "${TURBOQUANT_ROOT:?Export TURBOQUANT_ROOT before sbatch}"
: "${MODEL_ROOT:?Export MODEL_ROOT before sbatch}"
: "${OUTPUT_ROOT:?Export OUTPUT_ROOT before sbatch}"
: "${CASES_FILE:?Export CASES_FILE before sbatch}"
: "${RISK_RANKING:?Export RISK_RANKING before sbatch}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
CONFIG_FILE="${CONFIG_FILE:-configs/default_experiment.yaml}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_ROOT}/${MODEL_KEY}_full_pipeline.csv}"

cd "${TBGMP_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

test -f "${CONFIG_FILE}"
test -f "${CASES_FILE}"
test -f "${RISK_RANKING}"
test -d "${MODEL_ROOT}"
test -d "${TURBOQUANT_ROOT}"

"${PYTHON_BIN}" experiments/run_full_pipeline.py \
  --config "${CONFIG_FILE}" \
  --cases "${CASES_FILE}" \
  --model-key "${MODEL_KEY}" \
  --model-root "${MODEL_ROOT}" \
  --backend turboquant \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --risk-ranking "${RISK_RANKING}" \
  --maximum-topk 12 \
  --output "${OUTPUT_FILE}"
