#!/bin/bash

#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/llm-watermark-adaptive/log/download.%j.out
#SBATCH --error=/home/anthony.li/llm-watermark-adaptive/log/download.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-adaptive

export HF_HOME=/scratch/user/anthony.li/hf_cache

python 2-download.py
