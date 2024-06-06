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

# Initialize an empty array to hold node/cpu load pairs
parallel_node_list=()

# Iterate over each node assigned to the job
while read node; do
  # Extract the current CPU load from scontrol output
  cpus=$(scontrol show node "$node" | awk -F'=' '/CPULoad/ {gsub(/[^0-9.]/, "", $2); print $2}')

  # Append the node and its CPU load to the list
  parallel_node_list+=("${node}/${cpus}")
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

# Generate sshloginfile.txt without extra spaces
parallel_node_list_string=$(IFS=$'\n'; echo "${parallel_node_list[*]}" | awk -F'/' '{print "anthony.li@" $1 "#" $2}')
echo "$parallel_node_list_string" > sshloginfile.txt

# Output the result
echo "Parallel node list: $parallel_node_list_string"

# Run GNU Parallel with the list of nodes and their respective slots
/home/anthony.li/.conda/envs/watermark/bin/parallel \
  --sshloginfile sshloginfile.txt \
  --sshdelay 0.1 \
  -j $SLURM_NTASKS \
  --progress \
  bash ./detect-helper.sh {1} {2} {3} {4} \
  ::: gumbel \
  ::: $(seq 1 200) \
  ::: deletion insertion substitution \
  ::: 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
