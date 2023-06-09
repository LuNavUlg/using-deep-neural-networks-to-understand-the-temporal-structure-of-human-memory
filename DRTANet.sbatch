#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=DRTANet          # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=DRTANet.out   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=10G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=4:00:00              # Max execution time
#SBATCH --partition=quadro,tesla,a5000
#

# Activate your Anaconda environment
conda activate tfe # CHANGEME


# Time-series accumulation
python3 DRTANet_script.py --type series --videos 64

# Time-series accumulation
python3 DRTANet_script.py --type real_duration --videos 64