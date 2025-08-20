#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <model_name> <output_folder> <output_file> <task_name> <file_name> <comment>"
  exit 1
fi

MODEL_NAME=$1
OUTPUT_FOLDER=$2
OUTPUT_FILE=$3
TASK_NAME=$4
FILE_NAME=$5
COMMENT=$6
export TASK_NAME

TMP_SCRIPT=$(mktemp $(pwd)/slurm_job_XXXXXX.slurm)
cat <<EOL > $TMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=${COMMENT}_${TASK_NAME}
#SBATCH --output=$(pwd)/slurm_output/${COMMENT}_${TASK_NAME}.log
#SBATCH --error=$(pwd)/slurm_output/${COMMENT}_${TASK_NAME}.log
#SBATCH --time=24:00:00
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

# Replace MODEL_NAME / OUTPUT_FOLDER / OUTPUT_FILE in target script
grep -v "^#" scripts/${FILE_NAME}.sh | \
  sed "s/^MODEL_NAME=.*/MODEL_NAME=${MODEL_NAME}/" | \
  sed "s/^OUTPUT_FOLDER=.*/OUTPUT_FOLDER=${OUTPUT_FOLDER}/" | \
  sed "s/^OUTPUT_FILE=.*/OUTPUT_FILE=${OUTPUT_FILE}/" >> $TMP_SCRIPT

sbatch $TMP_SCRIPT
rm "$TMP_SCRIPT" 
