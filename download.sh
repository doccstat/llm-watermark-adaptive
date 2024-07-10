#!/bin/bash

#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/download.%j.out
#SBATCH --error=/home/anthony.li/out/download.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-adaptive

export HF_HOME=/scratch/user/anthony.li/hf_cache

python download.py
