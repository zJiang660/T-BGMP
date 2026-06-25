#!/usr/bin/env bash
# Stage A template. Replace placeholders and provide an external backend.
#SBATCH --partition=<a800-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage_a_%j.out
#SBATCH --error=logs/stage_a_%j.err

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
  --output "${OUTPUT_ROOT}/pipeline_results.csv"
