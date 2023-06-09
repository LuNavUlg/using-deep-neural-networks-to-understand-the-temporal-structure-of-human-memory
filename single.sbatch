#!/usr/bin/env bash
#
# Script arguments : 
# $1 : LASSO - naive - vanilla        
# $2 : Model to use                   (alexnet-resnet18-efficientnetB0-efficientnetB4-efficientnetB7)
# $3 : Number of videos               (int)
# $4 : Compute accumulators           (true-false)
#
# Slurm arguments
# ---------------------------------------------------------------------------------
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=2:00:00              # Max execution time
# ---------------------------------------------------------------------------------


# To remove opencv logs
export OPENCV_LOG_LEVEL=OFF

# Activate your Anaconda environment
conda activate tfe # CHANGEME

# The python script has the following arguments:
# -m : Model to use                         (alexnet - resnet)
# -v : Number of videos to use
# -t : Type of execution                    (vanilla - naive - lasso)
# -f : max number of features to use
# --p : plot the results                    (default yes)
# --c : calculate accumulators              (default yes)


# NAIVE FEATURE SELECTION
if [ $1 = "naive" ]  
then
    if [ $4 = true ]  
    then
        python script.py -m $2 -v $3 -t $1 -f 2 --id $SLURM_ARRAY_TASK_ID --c --p
    else
        python script.py -m $2 -v $3 -t $1 -f 2 --id $SLURM_ARRAY_TASK_ID --no-c --p
    fi
        
# LASSO AND VANILLA
else
    if [ $4 = true ]  
    then
        python script.py -m $2 -v $3 -t $1 --c --p
    else
        python script.py -m $2 -v $3 -t $1 --no-c --p
    fi
fi