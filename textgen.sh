#!/bin/bash

# Provide a name for your job, so it may be recognized in the output of squeue
# SBATCH --job-name=textgen

# Define how many nodes this job needs.
# This example uses one 1 node.  Recall that each node has 128 CPU cores.
#SBATCH --nodes=1

#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Define a maximum amount of time the job will run in real time. This is a hard
# upper bound, meaning that if the job runs longer than what is written here, it
# will be terminated by the server.
#              d-hh:mm:ss
#SBATCH --time=7-00:00:00

# Define the partition on which the job shall run.
##SBATCH --partition=gpu
#SBATCH --partition=long,xlong

# Define how much memory you need. Choose one of the following:
# --mem will define memory per node and
# --mem-per-cpu will define memory per CPU/core.
##SBATCH --mem-per-cpu=1024MB
#SBATCH --mem=128GB        # The double hash means that this one is not in effect

# Define any general resources required by this job.  In this example 1 "a30"
# GPU is requested per node.  Note that gpu:1 would request any gpu type, if
# available.  This cluster currenlty only contains NVIDIA A30 GPUs.
##SBATCH --gres=gpu:a30:1

# Define the destination file name(s) for this batch scripts output.
# The use of '%j' here uses the job ID as part of the filename.
#SBATCH --output=/home/anthony.li/out/textgen.%j.out
#SBATCH --error=/home/anthony.li/out/textgen.%j.err

# Turn on mail notification. There are many possible values, and more than one
# may be specified (using comma separated values):
# NONE, BEGIN, END, FAIL, REQUEUE, ALL, INVALID_DEPEND, STAGE_OUT, TIME_LIMIT,
# TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50 - See "man sbatch" or the slurm
# website for more values (https://slurm.schedmd.com/sbatch.html).
#SBATCH --mail-type=ALL

# The email address to which emails should be sent.
#SBATCH --mail-user=anthony.li@tamu.edu

# All commands should follow the last SBATCH directive.

# Define or set any necessary environment variables for this job.
# Note that several environment variables have been defined for you, and two
# of particular interest are:
#   SCRATCH=/scratch/user/NetID  # This folder is accessible from any node.
#   TMPDIR=/tmp/job.%j  # This folder is automatically created / destroyed for
#                       # you at the start / end of each job. This folder exists
#                       # locally on a compute node using a fast local disk.  It
#                       # is not directly accessible from any other node.

# As an example, if your application requires the loading of many files it may
# be faster, and certainly more efficient, to first copy those files to TMPDIR.
# Doing so ensures that the files are copied across the network once, and are
# accessible to the application locally on each node using a fast disk.
# cp watermark/demo/* ${TMPDIR}

# Load any modules that are required.  Note that while the system does provide a
# default set of basic tools, it does not include all of the software you will
# need for your job.  As such you should specify the modules for the software
# packages and versions that your job needs here.
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
  for pcts in 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    # for attack in deletion insertion substitution; do
    #   python textgen.py --save results/gpt-$method-$attack-$n-$m-$pcts.p --n $n --batch_size 100 --m $m --buffer_tokens 0 --model openai-community/gpt2 --seed 1 --T 1000 --method $method --${attack} $pcts
    # done
    # for attack in deletion insertion substitution; do
    #   python textgen.py --save results/opt-$method-$attack-$n-$m-$pcts.p --n $n --batch_size 100 --m $m --buffer_tokens 0 --model facebook/opt-1.3b --seed 1 --T 1000 --method $method --${attack} $pcts
    # done
    # no gpu for llama
    for attack in deletion insertion substitution; do
      python textgen.py --save results/ml3-$method-$attack-$n-$m-$pcts.p --n $n --batch_size 100 --m $m --buffer_tokens 0 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 1000 --method $method --${attack} $pcts
    done
  done
done
