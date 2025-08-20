#!/bin/bash


# keywords
COMMENT="all"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_keywords scienceqa_qs_expr keywords_${COMMENT} key_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="1k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_keywords_${COMMENT} scienceqa_qs_expr keywords_${COMMENT} key_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="2k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_keywords_${COMMENT} scienceqa_qs_expr keywords_${COMMENT} key_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="3k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_keywords_${COMMENT} scienceqa_qs_expr keywords_${COMMENT} key_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="4k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_keywords_${COMMENT} scienceqa_qs_expr keywords_${COMMENT} key_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="5k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_keywords_${COMMENT} scienceqa_qs_expr keywords_${COMMENT} key_eval scienceqa/run_eval_keywords "${COMMENT}"

# normal
COMMENT="all"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_normal_v2 scienceqa_qs_expr normal_${COMMENT} normal_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="1k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_normal_v2_${COMMENT} scienceqa_qs_expr normal_${COMMENT} normal_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="2k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_normal_v2_${COMMENT} scienceqa_qs_expr normal_${COMMENT} normal_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="3k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_normal_v2_${COMMENT} scienceqa_qs_expr normal_${COMMENT} normal_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="4k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_normal_v2_${COMMENT} scienceqa_qs_expr normal_${COMMENT} normal_eval scienceqa/run_eval_keywords "${COMMENT}"

COMMENT="5k"
bash scripts/run_slurm_umiacs.sh qwen25_3b_scienceqa_lora_scienceqa_normal_v2_${COMMENT} scienceqa_qs_expr normal_${COMMENT} normal_eval scienceqa/run_eval_keywords "${COMMENT}"

