#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <dataset> <task_name> <file_name>"
  exit 1
fi

FILE_NAME=$1
TUNE_LLM=$2
TUNE_VISION=$3
TUNE_MLP=$4

TMP_SCRIPT=$(mktemp $(pwd)/slurm_job_XXXXXX.slurm)
cat <<EOL > $TMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=${FILE_NAME}_llm${TUNE_LLM}_mlp${TUNE_MLP}_vision${TUNE_VISION}
#SBATCH --output=$(pwd)/slurm_output/${FILE_NAME}_llm${TUNE_LLM}_mlp${TUNE_MLP}_vision${TUNE_VISION}.log
#SBATCH --error=$(pwd)/slurm_output/${FILE_NAME}_llm${TUNE_LLM}_mlp${TUNE_MLP}_vision${TUNE_VISION}.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

cd $(pwd)

EOL

if [ ! -d "$(pwd)/slurm_output" ]; then
  mkdir -p "$(pwd)/slurm_output"
fi


grep -v "^#" scripts/colorbench/${FILE_NAME}.sh \
  | sed -e "s/TUNE_LLM=.*/TUNE_LLM=${TUNE_LLM}/" \
        -e "s/TUNE_MLP=.*/TUNE_MLP=${TUNE_MLP}/" \
        -e "s/TUNE_VISION=.*/TUNE_VISION=${TUNE_VISION}/" \
  >> $TMP_SCRIPT

sbatch $TMP_SCRIPT

rm "$TMP_SCRIPT" 
