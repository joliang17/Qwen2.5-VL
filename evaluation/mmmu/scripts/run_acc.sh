#!/bin/bash

#SBATCH --job-name=qwen_infer
#SBATCH --output=qwen_infer.log
#SBATCH --error=qwen_infer.log
#SBATCH --time=24:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1

RESULT_TYPE="mcq"  # mcq / normal / baseline
INPUT_FILE="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mcq_acc/${RESULT_TYPE}_ver.json"
OUTPUT_FOLDER="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mcq_parsed"

python3 run_scienceqa_v2.py eval --input-file="${INPUT_FILE}" --output-folder="${OUTPUT_FOLDER}" --api-key="${API_KEY}" --result-type="${RESULT_TYPE}"

