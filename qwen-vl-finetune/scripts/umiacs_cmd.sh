#!/bin/bash

# # keywords
# bash scripts/run_slurm_umiacs.sh scienceqa_keywords_1k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_keywords_2k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_keywords_3k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_keywords_4k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_keywords_5k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_keywords lora scienceqa/sft_1gpu_lora_para

# # normal
# bash scripts/run_slurm_umiacs.sh scienceqa_normal_v2_1k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_normal_v2_2k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_normal_v2_3k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_normal_v2_4k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_normal_v2_5k lora scienceqa/sft_1gpu_lora_para
# bash scripts/run_slurm_umiacs.sh scienceqa_normal_v2 lora scienceqa/sft_1gpu_lora_para

# sbatch scripts/colorbench/sft_1gpu_colorbench_lora_llm.sh
# sbatch scripts/colorbench/sft_1gpu_colorbench_lora_llm_vision.sh
# sbatch scripts/colorbench/sft_1gpu_colorbench_lora_llm_mlp.sh
# sbatch scripts/colorbench/sft_1gpu_colorbench_lora_llm_vision_mlp.sh


# # ADDED: bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para llm vision mlp

bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para False False True  # mlp
bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para False True False  # vision
bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para False True True  # vision + mlp
bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para True False False  # llm
bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para True False True  # llm + mlp
bash scripts/run_slurm_umiacs_colorbench.sh 3b_ft_para True True True  # llm + vision + mlp


bash scripts/run_slurm_umiacs_colorbench.sh 3b_lora_para True False False  # llm
bash scripts/run_slurm_umiacs_colorbench.sh 3b_lora_para True False True  # llm + mlp
bash scripts/run_slurm_umiacs_colorbench.sh 3b_lora_para True True False  # llm + vision
bash scripts/run_slurm_umiacs_colorbench.sh 3b_lora_para True True True  # llm + vision + mlp
