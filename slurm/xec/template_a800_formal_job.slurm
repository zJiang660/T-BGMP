#!/usr/bin/env bash
# Sanitized XEC A800 template for reproducing formal T-BGMP jobs.
# Replace TBGMP_HPC_PROJECT with the project directory on the cluster.
# Raw logs should remain on the cluster and should not be committed.
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/formal_a800_%j.out
#SBATCH --error=logs/formal_a800_%j.err

set -euo pipefail

: "${TBGMP_HPC_PROJECT:?Set TBGMP_HPC_PROJECT to the cluster project path}"
cd "${TBGMP_HPC_PROJECT}"
source scripts/activate_tbgmp_env.sh

# Replace this command for a specific formal experiment.
# "${TBGMP_PYTHON}" scripts/your_formal_experiment.py --resume
