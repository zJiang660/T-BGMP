#!/usr/bin/env bash
# Sanitized template, not a turn-key cluster job. Adapt partition/account
# names, environment modules, resources, and paths for the target HPC system.
#SBATCH --partition=<a800-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage_a_%j.out
#SBATCH --error=logs/stage_a_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/T-BGMP
MODEL_ROOT=/path/to/models
OUTPUT_ROOT=/path/to/outputs
MODEL_PATH="${MODEL_ROOT}/model"
CASES_FILE=/path/to/cases.csv
BACKEND_MODULE=your_backend.module:create_backend

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs
python experiments/run_full_pipeline.py \
  --cases "${CASES_FILE}" \
  --model-path "${MODEL_PATH}" \
  --model-id example-model \
  --backend "${BACKEND_MODULE}" \
  --output "${OUTPUT_ROOT}/pipeline_results.csv"
