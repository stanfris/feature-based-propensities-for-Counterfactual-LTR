#!/bin/bash

# Parameters
#SBATCH --array=0-287%288
#SBATCH --cpus-per-task=16
#SBATCH --error=/gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/position_bias_models/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --job-name=run
#SBATCH --mem=120GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/position_bias_models/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=fat_rome
#SBATCH --signal=USR2@120
#SBATCH --time=400
#SBATCH --wckey=submitit

# setup
module purge
module load 2023
module load CUDA/12.4.0
source .venv/bin/activate

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/position_bias_models/.submitit/%A_%a/%A_%a_%t_log.out --error /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/position_bias_models/.submitit/%A_%a/%A_%a_%t_log.err /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/.venv/bin/python -u -m submitit.core._submit /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/position_bias_models/.submitit/%j
