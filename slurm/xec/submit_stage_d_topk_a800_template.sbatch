#!/usr/bin/env bash
# Top-k recovery template. The runner performs Stage A first, mines sensitive
# cases from found fields, then executes Top1--Top12 and matched controls.
#SBATCH --partition=<a800-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage_d_topk_%j.out
#SBATCH --error=logs/stage_d_topk_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/T-BGMP
MODEL_PATH=/path/to/models/model
OUTPUT_ROOT=/path/to/outputs
BACKEND_MODULE=your_backend.module:create_backend

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs
python experiments/run_full_pipeline.py \
  --cases /path/to/cases.csv \
  --model-path "${MODEL_PATH}" \
  --model-id example-model \
  --backend "${BACKEND_MODULE}" \
  --risk-ranking /path/to/risk_ranking.csv \
  --maximum-topk 12 \
  --output "${OUTPUT_ROOT}/pipeline_results.csv"
