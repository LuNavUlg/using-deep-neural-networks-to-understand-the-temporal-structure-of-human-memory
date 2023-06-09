#!/usr/bin/env bash
#
# Script arguments : 
# $1 : Number of videos              (int)
# $2 : Compute accumulators          (true-false)
# $3 : Calibrate layers              (fixed-stats)
#
# Slurm arguments
# ---------------------------------------------------------------------------------
#SBATCH --job-name=all          # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=all.log            # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=10:00:00              # Max execution time
#
# ---------------------------------------------------------------------------------


conda activate tfe
cd atfe

# Calibrate parameters
sbatch --wait calibrate.sbatch $3

# Baseline: DRTANet
sbatch DRTANet.sbatch

# Baseline: Dummies
python3 baseline.py --videos 64 --type dummy
python3 baseline.py --videos 64 --type real_duration

# -------------------------------------- VANILLA --------------------------------------
# ALEXNET 
sbatch  --job-name=exp1 --output=exp1/%x.log --wait single.sbatch vanilla alexnet $1 $2

# --------------------------------------- LASSO ---------------------------------------
# ALEXNET 
sbatch --job-name=exp2 --output=exp2/%x.log single.sbatch lasso alexnet $1 false 
# RESNET
sbatch --job-name=exp4 --output=exp4/%x.log single.sbatch lasso resnet18 $1 $2 
# EFFICIENTNETB0
sbatch --job-name=exp6 --output=exp6/%x.log --wait single.sbatch lasso efficientnetB0 $1 $2 
# EFFICIENTNETB4
sbatch --job-name=exp8 --output=exp8/%x.log --wait single.sbatch lasso efficientnetB4 $1 $2 
# EFFICIENTNETB7
sbatch --job-name=exp10 --output=exp10/%x.log --wait single.sbatch lasso efficientnetB7 $1 $2 

# ------------------------------ NAIVE FEATURE SELECTION ------------------------------
# ALEXNET 
sbatch --job-name=exp3 --array=0-65 --output=exp3/%x-%a.log single.sbatch naive alexnet $1 false
# RESNET
sbatch --job-name=exp5 --array=0-27 --output=exp5/%x-%a.log single.sbatch naive resnet18 $1 false
# EFFICIENTNETB0
sbatch --job-name=exp7 --array=0-65 --output=exp7/%x-%a.log single.sbatch naive efficientnetB0 $1 false
# EFFICIENTNETB4
sbatch --job-name=exp9 --array=0-65 --output=exp9/%x-%a.log single.sbatch naive efficientnetB4 $1 false
# EFFICIENTNETB7
sbatch --job-name=exp11 --array=0-65 --output=exp11/%x-%a.log single.sbatch naive efficientnetB7 $1 false
        





