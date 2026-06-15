#!/bin/bash

# Parameters
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=48
#SBATCH --error=/gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/true_baselines/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --job-name=run
#SBATCH --mem=84GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/true_baselines/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=rome
#SBATCH --signal=USR2@120
#SBATCH --time=100
#SBATCH --wckey=submitit

# setup
module purge
module load 2023
module load CUDA/12.4.0
source .venv/binactivate

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/true_baselines/.submitit/%A_%a/%A_%a_%t_log.out --error /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/true_baselines/.submitit/%A_%a/%A_%a_%t_log.err /gpfs/home5/sfris1/two-towers-confounding-project/.venv/bin/python -u -m submitit.core._submit /gpfs/home5/sfris1/Doubly-Robust-Optimization-With-Two-Tower-Models/results/true_baselines/.submitit/%j
