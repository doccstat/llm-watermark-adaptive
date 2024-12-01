#!/bin/bash

# Determine the total number of commands by counting lines in 3-detect-commands.sh
total_commands=$(wc -l < 3-detect-commands.sh)

# The total number of concurrently running jobs. This number should be the
# smaller of the maximum array id and the total number of commands.
total_jobs=1000

# Calculate the number of commands per job (minimum)
commands_per_job=$((total_commands / total_jobs))

# Calculate the number of jobs that need to process an extra command
extra_commands=$((total_commands % total_jobs))

# Loop through each task ID and extract the corresponding command
for task_id in XXX YYY ZZZ; do
  # Determine the start and end command index for this particular job
  if [ "$task_id" -le "$extra_commands" ]; then
    start_command=$(( (task_id - 1) * (commands_per_job + 1) + 1 ))
    end_command=$(( task_id * (commands_per_job + 1) ))
  else
    start_command=$(( extra_commands * (commands_per_job + 1) + (task_id - extra_commands - 1) * commands_per_job + 1 ))
    end_command=$(( extra_commands * (commands_per_job + 1) + (task_id - extra_commands) * commands_per_job ))
  fi

  # Use sed to extract the range of commands in one operation and append to the rerun file
  sed -n "${start_command},${end_command}p" 3-detect-commands.sh >> 3-detect-commands-rerun.sh
done

mv 3-detect-commands-rerun.sh 3-detect-commands.sh
