#!/usr/bin/env bash
#
# Script arguments : 
# $1 : Type of calibration (fixed or stats)
#
# Slurm arguments
# ---------------------------------------------------------------------------------
#SBATCH --job-name=calibrate          # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=calibrate.log            # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=2:00:00              # Max execution time
#
# ---------------------------------------------------------------------------------

conda activate tfe
cd atfe

# Calibrate parameters
python3 calibration.py --type $1