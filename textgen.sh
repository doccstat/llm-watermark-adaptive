#!/bin/bash

#SBATCH --job-name=textgen
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1

##SBATCH --time=7-00:00:00
##SBATCH --partition=long,xlong

#SBATCH --mem=128GB
#SBATCH --output=/home/anthony.li/out/textgen.%j.out
#SBATCH --error=/home/anthony.li/out/textgen.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-adaptive

mkdir -p results
mkdir -p log

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo $(which python)

export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

n=20
m=20

for method in gumbel; do
  for pcts in 0.0 0.05 0.1 0.2 0.3; do
    for attack in deletion insertion substitution; do
      python textgen.py --save results/gpt-$method-$attack-$n-$m-$pcts.p --n $n --batch_size 100 --m $m --buffer_tokens 0 --model openai-community/gpt2 --seed 1 --T 1000 --method $method --${attack} $pcts --candidate_prompt_max 10
    done
    for attack in deletion insertion substitution; do
      python textgen.py --save results/opt-$method-$attack-$n-$m-$pcts.p --n $n --batch_size 100 --m $m --buffer_tokens 0 --model facebook/opt-1.3b --seed 1 --T 1000 --method $method --${attack} $pcts --candidate_prompt_max 10
    done
    # no gpu for llama
    # for attack in deletion insertion substitution; do
    #   python textgen.py --save results/ml3-$method-$attack-$n-$m-$pcts.p --n $n --batch_size 100 --m $m --buffer_tokens 0 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 1000 --method $method --${attack} $pcts --candidate_prompt_max 10
    # done
  done
done
