#!/bin/bash

#SBATCH --job-name=textgen
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,xgpu
#SBATCH --gres=gpu:a30:2

#SBATCH --mem=64GB
#SBATCH --output=/home/anthony.li/out/textgen.%A.%a.out
#SBATCH --error=/home/anthony.li/out/textgen.%A.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-3

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-adaptive

mkdir -p results
mkdir -p log

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo $(which python)

export PATH="/home/anthony.li/.local/bin:$PATH"
export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

# Define attack values for each array task
declare -a attacks=("deletion" "insertion" "substitution")

# Fetches the correct attack type based on SLURM_ARRAY_TASK_ID
attack=${attacks[$SLURM_ARRAY_TASK_ID-1]}

for watermark_key_length in 20 50 80 100 500 1000; do
  # Set tokens_count based on watermark_key_length
  if [ $watermark_key_length -le 100 ]; then
    tokens_count=$watermark_key_length
  else
    tokens_count=100
  fi

  for method in gumbel; do
    for pcts in 0.0 0.05 0.1 0.2 0.3; do
      python 3-textgen.py --save results/gpt-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 100 --tokens_count $tokens_count --buffer_tokens 0 --model openai-community/gpt2 --seed 1 --T 100 --method $method --${attack} $pcts --candidate_prompt_max 10
      # python 3-textgen.py --save results/opt-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 100 --tokens_count $tokens_count --buffer_tokens 0 --model facebook/opt-1.3b --seed 1 --T 100 --method $method --${attack} $pcts --candidate_prompt_max 10
      python 3-textgen.py --save results/ml3-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 100 --tokens_count $tokens_count --buffer_tokens 0 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 100 --method $method --${attack} $pcts --candidate_prompt_max 10
    done
  done
done
