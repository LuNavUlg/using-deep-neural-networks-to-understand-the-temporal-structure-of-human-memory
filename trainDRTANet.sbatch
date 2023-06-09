#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=trainDR-TANet          # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=trainDR-TANet.out   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=10G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:4                # Number of GPU's
#SBATCH --time=24:00:00              # Max execution time
#SBATCH --partition=quadro,tesla,a5000
#

# export PYTORCH_CUDA_ALLOC_CONF=MAX_SPLIT_SIZE_MB=1024

# Activate your Anaconda environment
conda activate tfe # CHANGEME

python3 DR-TANet/train.py --dataset vl_cmu_cd --datadir vl_cmu_cd_binary_mask --checkpointdir checkpoints_DR_TANet --max-epochs 150 --batch-size 8 --encoder-arch resnet18 --epoch-save 25 --drtam --refinement