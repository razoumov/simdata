#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7200
#SBATCH --time=3:0:0
#SBATCH --account=def-some-user
source ~/env-jax/bin/activate
python train.py
