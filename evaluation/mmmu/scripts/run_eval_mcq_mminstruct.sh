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

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1


MODEL_NAME="qwen25_3b_mminstruct_lora_keywords_1e-4"
MODEL_PATH="/fs/nexus-projects/wilddiffusion/vlm/qwen_mcq/${MODEL_NAME}"

# DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mminstruct/test_normal.json"
# OUTPUT_PATH="mminstruct_lora_1e4/keywords_onqa_ver.json"
# python3 run_scienceqa_v2_filterwords.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="mminstruct" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"


DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mminstruct/test_keywords_samples_ori.json"
OUTPUT_PATH="mminstruct_lora_1e4_samples_ori/keywords_ver.json"
# python3 run_scienceqa_v2_filterwords.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="mminstruct" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"

python3 qs_similarity.py --json_path="${OUTPUT_PATH}"
