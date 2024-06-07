#!/bin/bash

#SBATCH --job-name=detect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/detect.%A.%a.out
#SBATCH --error=/home/anthony.li/out/detect.%A.%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-1000

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-adaptive

OFFSET=3000

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

# Adjust the task ID calculation
ACTUAL_TASK_ID=$((${SLURM_ARRAY_TASK_ID} + OFFSET))

command=$(sed -n "${ACTUAL_TASK_ID}p" detect-commands.sh)

echo "Running task with ACTUAL_TASK_ID = $ACTUAL_TASKID"
echo "Executing: $command"

eval $command
