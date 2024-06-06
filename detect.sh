#!/bin/bash

#SBATCH --job-name=detect
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/detect.%j.%a.out
#SBATCH --error=/home/anthony.li/out/detect.%j.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-adaptive

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" detect-commands.sh)

echo "Running task with SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Executing: $command"

eval $command
