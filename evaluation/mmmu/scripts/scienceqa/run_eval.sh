#!/bin/bash

#SBATCH --job-name=qwen_mcq_eval
#SBATCH --output=qwen_mcq_eval.log
#SBATCH --error=qwen_mcq_eval.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# Print job info (goes into .out log file)
echo "[INFO] SLURM job started"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node list: $SLURM_NODELIST"
echo "[INFO] GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------"

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1


MODEL_NAME="qwen25_3b_scienceqa_key_lora_keywords_1e-4"
OUTPUT_FOLDER="scienceqa_key_lora_1e4"
OUTPUT_FILE="keywords_onqa_ver"

MODEL_PATH="/fs/nexus-projects/wilddiffusion/vlm/qwen_mcq/${MODEL_NAME}"
DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/scienceqa/normal.json"
OUTPUT_PATH="${OUTPUT_FOLDER}/${OUTPUT_FILE}.json"
python3 run_scienceqa_v2.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="scienceqa" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"

