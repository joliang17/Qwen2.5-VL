#!/bin/bash

#SBATCH --job-name=qwen_normal_eval
#SBATCH --output=qwen_normal_eval.log
#SBATCH --error=qwen_normal_eval.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1

MODEL_PATH="/fs/nexus-projects/wilddiffusion/vlm/qwen/qwen25_checkpoints_normal_3epochs"
DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/scienceqa/normal.json"
OUTPUT_PATH="mcq_acc_3epochs/normal_ver.json"

# python3 run_scienceqa.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="scienceqa" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"
python3 run_scienceqa_v2.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="scienceqa" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"

