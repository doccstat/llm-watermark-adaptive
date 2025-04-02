#!/bin/bash

rm -f 3-textgen-commands.sh

for model_prefix in ml3 mt7; do
  if [ "$model_prefix" = "ml3" ]; then
    model="meta-llama/Meta-Llama-3-8B"
  else
    model="mistralai/Mistral-7B-v0.1"
  fi
  for watermark_key_length in 10 20 30; do
    for attack in deletion insertion substitution; do
      if [ "$attack" = "substitution" ]; then
        pcts_list=(0.0 0.1 0.2 0.3 1.0)
      elif [ "$attack" = "deletion" ]; then
        pcts_list=(1.0)
      else
        pcts_list=(1.0)
      fi
      tokens_count=$watermark_key_length
      for method in gumbel transform; do
        for pcts in "${pcts_list[@]}"; do
          echo "python 3-textgen.py --save results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts --watermark_key_length $watermark_key_length --batch_size 25 --tokens_count $tokens_count --buffer_tokens 0 --model $model --seed 1 --T 1000 --method $method --${attack} $pcts" >> 3-textgen-commands.sh
        done
      done
    done
  done
done
