#!/bin/bash

rm -f 4-detect-commands.sh

for k_tokens_count_ratio in 0.3 0.6 1.0; do
  for model_prefix in ml3 mt7; do
    if [ "$model_prefix" = "ml3" ]; then
      model="meta-llama/Meta-Llama-3-8B"
    else
      model="mistralai/Mistral-7B-v0.1"
    fi

    for watermark_key_length in 10 20 30 40 50; do
      for attack in deletion insertion substitution; do
        if [ "$attack" = "substitution" ]; then
          pcts_list=(0.0 0.1 0.2 0.3)
        else
          pcts_list=(1.0)
        fi

        tokens_count=$watermark_key_length

        k=$(awk "BEGIN {print int($tokens_count * $k_tokens_count_ratio)}")

        for method in gumbel transform; do
          for pcts in "${pcts_list[@]}"; do
            rm -rf results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts-$k-detect
            mkdir -p results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts-$k-detect

            for Tindex in $(seq 0 99); do
              echo "python 4-detect.py --token_file results/${model_prefix}-${method}-${attack}-${watermark_key_length}-${tokens_count}-${pcts} --n ${watermark_key_length} --model ${model} --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --n_runs 999" >> 4-detect-commands.sh
            done
          done
        done
      done
    done
  done
done
