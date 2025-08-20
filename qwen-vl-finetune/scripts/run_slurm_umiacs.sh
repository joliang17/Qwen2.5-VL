#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <dataset> <task_name> <file_name>"
  exit 1
fi

DATASET=$1
TASK_NAME=$2
FILE_NAME=$3
export TASK_NAME

TMP_SCRIPT=$(mktemp $(pwd)/slurm_job_XXXXXX.slurm)
cat <<EOL > $TMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=${DATASET}_${TASK_NAME}
#SBATCH --output=$(pwd)/slurm_output/${DATASET}_${TASK_NAME}.log
#SBATCH --error=$(pwd)/slurm_output/${DATASET}_${TASK_NAME}.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

cd $(pwd)

EOL

if [ ! -d "$(pwd)/slurm_output" ]; then
  mkdir -p "$(pwd)/slurm_output"
fi

grep -v "^#" scripts/${FILE_NAME}.sh | sed "s/DATASET=.*$/DATASET=${DATASET}/" >> $TMP_SCRIPT

sbatch $TMP_SCRIPT

rm "$TMP_SCRIPT" 
