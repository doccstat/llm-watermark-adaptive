#!/bin/bash

#SBATCH --job-name=detect
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/detect.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-adaptive

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo $(which python)

# Create directories for results
for method in gumbel; do
  for model in opt gpt; do
    for pcts in 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
      for attack in deletion insertion substitution; do
        mkdir -p results/$model-$method-$attack-10-10-$pcts.p-detect
      done
    done
  done
done

expanded_nodes=$(scontrol show hostname $SLURM_JOB_NODELIST | tr '\n' ',')
expanded_nodes=${expanded_nodes%?}

echo "Expanded nodes: $expanded_nodes"

# Create a GNU Parallel-compatible list of nodes with the number of slots (CPUs) per node
parallel_node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST | while read node; do echo -n "${node}/$(sinfo --exact -o "%C" -n ${node} | grep -oP '\d+(?=/\d+/\d+/)') "; done)

echo "Parallel node list: $parallel_node_list"

# Run GNU Parallel with the list of nodes and their respective slots
/home/anthony.li/.conda/envs/watermark/bin/parallel \
  --sshloginfile <(echo "$parallel_node_list") \
  -j $SLURM_NTASKS \
  --progress \
  bash ./detect-helper.sh {1} {2} {3} {4} \
  ::: gumbel \
  ::: $(seq 1 200) \
  ::: deletion insertion substitution \
  ::: 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
