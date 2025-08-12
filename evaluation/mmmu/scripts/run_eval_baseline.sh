#!/bin/bash

#SBATCH --job-name=qwen_baseline_eval
#SBATCH --output=qwen_baseline_eval.log
#SBATCH --error=qwen_baseline_eval.log
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

DATA_PATH="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mminstruct/test_keywords.json"


MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_PATH="qs_gene/3B_ver.json"
python3 run_scienceqa_v2.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="scienceqa" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_PATH="qs_gene/7B_ver.json"
python3 run_scienceqa_v2.py infer --model-path="${MODEL_PATH}" --dataset-path="${DATA_PATH}" --dataset="scienceqa" --data-dir="${DATA_PATH}" --output-file="${OUTPUT_PATH}"


