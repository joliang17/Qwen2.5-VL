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

MODEL_NAME="qwen25_3b_mminstruct_lora_normal_1e-4"
MODEL_PATH="/fs/nexus-projects/wilddiffusion/vlm/qwen_mcq/${MODEL_NAME}"

DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mminstruct/test_normal.json"
OUTPUT_PATH="mminstruct_lora_1e4/normal_ver.json"
python3 run_scienceqa_v2.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="mminstruct" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"


DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mminstruct/test_keywords.json"
OUTPUT_PATH="mminstruct_lora_1e4/qa_onkeyword_ver.json"
python3 run_scienceqa_v2.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="mminstruct" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"

