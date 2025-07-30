#!/bin/bash

#SBATCH --job-name=post_process
#SBATCH --output=post_process.log
#SBATCH --error=post_process.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# data preparation
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1


# python3 data_prep_colorbench.py

# 7b model
sbatch scripts/sft_7b_colorbench_vision.sh
sbatch scripts/sft_7b_colorbench_mlp.sh
sbatch scripts/sft_7b_colorbench_llm_mlp.sh


# 3b model
sbatch scripts/sft_1gpu_colorbench_vision.sh
sbatch scripts/sft_1gpu_colorbench_mlp.sh
sbatch scripts/sft_1gpu_colorbench_llm_mlp.sh