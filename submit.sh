#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7200
#SBATCH --time=12:0:0
#SBATCH --account=def-razoumov-ac
#SBATCH --gpus-per-node=h100:1
#SBATCH --reservation=gpu_mig_test
#SBATCH --output=solution.out
source ~/env-jax/bin/activate
python train.py
