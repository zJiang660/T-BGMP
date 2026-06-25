#!/usr/bin/env bash
# Sanitized XEC A800 discovery submission.
# Replace TBGMP_HPC_PROJECT with the project directory on the cluster.
# Raw logs should remain on the cluster and should not be committed.
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/discovery_all_models_a800_%j.out
#SBATCH --error=logs/discovery_all_models_a800_%j.err

set -euo pipefail

: "${TBGMP_HPC_PROJECT:?Set TBGMP_HPC_PROJECT to the cluster project path}"
cd "${TBGMP_HPC_PROJECT}"
source scripts/activate_tbgmp_env.sh
"${TBGMP_PYTHON}" scripts/run_discovery_all_models_formal.py --resume
