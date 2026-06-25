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

export TBGMP_ROOT=/path/to/T-BGMP
export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
CASES_FILE=/path/to/cases.csv

cd "${TBGMP_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs
python experiments/run_full_pipeline.py \
  --config configs/default_experiment.yaml \
  --cases "${CASES_FILE}" \
  --model-key qwen25_3b \
  --model-root "${MODEL_ROOT}" \
  --backend turboquant \
  --turboquant-root "${TURBOQUANT_ROOT}" \
  --output "${OUTPUT_ROOT}/pipeline_results.csv"
