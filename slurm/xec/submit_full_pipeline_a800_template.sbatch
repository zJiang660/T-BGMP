#!/usr/bin/env bash
# Generic template only. Replace placeholder paths for the target cluster.
# The minimal repository does not include the external model runner.
#SBATCH --partition=<a800-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tbgmp_pipeline_%j.out
#SBATCH --error=logs/tbgmp_pipeline_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/T-BGMP
MODEL_ROOT=/path/to/models
OUTPUT_ROOT=/path/to/outputs
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs

# Replace with a compatible external model-execution entry point.
"${PYTHON_BIN}" /path/to/external_runner.py \
  --project-root "${PROJECT_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --resume
