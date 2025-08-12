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

# Load API_KEY from key.conf
source /fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/config/key.conf
ROOT_FOLDER="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/"

# INPUT_FOLDER="mminstruct_scienceqa_2e5"
INPUT_FOLDER="scienceqa_key_lora_1e4"
RESULT_TYPE="keywords_onqa"  # mcq / normal / baseline
INPUT_FILE="${ROOT_FOLDER}/${INPUT_FOLDER}/${RESULT_TYPE}_ver.json"
OUTPUT_FOLDER="${ROOT_FOLDER}/${INPUT_FOLDER}_acc"

python3 run_scienceqa_v2.py eval --input-file="${INPUT_FILE}" --output-folder="${OUTPUT_FOLDER}" --api-key="${API_KEY}" --result-type="${RESULT_TYPE}"

# RESULT_TYPE="normal"  # mcq / normal / baseline
# INPUT_FILE="${ROOT_FOLDER}/${INPUT_FOLDER}/${RESULT_TYPE}_ver.json"
# OUTPUT_FOLDER="${ROOT_FOLDER}/${INPUT_FOLDER}_acc"

# python3 run_scienceqa_v2.py eval --input-file="${INPUT_FILE}" --output-folder="${OUTPUT_FOLDER}" --api-key="${API_KEY}" --result-type="${RESULT_TYPE}"
