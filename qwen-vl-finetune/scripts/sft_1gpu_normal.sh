#!/bin/bash

#SBATCH --job-name=qwen_normal
#SBATCH --output=qwen_normal.log
#SBATCH --error=qwen_normal.log
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


# Distributed training configuration
TRITON_CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=1

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm="Qwen/Qwen2.5-VL-3B-Instruct"  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=scienceqa_normal

# Output configuration
run_name="qwen2vl-baseline"
output_dir="/fs/nexus-projects/wilddiffusion/vlm/qwen/qwen25_checkpoints_normal_3epochs"
cache_dir="/fs/nexus-projects/wilddiffusion/cache"  

# Training arguments
    # --deepspeed ${deepspeed} \
args="
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --output_dir ${output_dir} \
    --cache_dir ${cache_dir} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

# python3 qwenvl/train/train_qwen.py  ${args}