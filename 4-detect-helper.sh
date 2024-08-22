#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

model_prefix=$1
method=$2
Tindex=$3
attack=$4
pcts=$5

if [ "$model_prefix" == "opt" ]; then
    model="facebook/opt-1.3b"
elif [ "$model_prefix" == "gpt" ]; then
    model="openai-community/gpt2"
elif [ "$model_prefix" == "ml3" ]; then
    model="meta-llama/Meta-Llama-3-8B"
elif [ "$model_prefix" == "mt7" ]; then
    model="mistralai/Mistral-7B-v0.1"
else
    echo "Invalid model prefix"
    exit 1
fi

n=20
m=20
k=10

python 4-detect.py --token_file "results/${model_prefix}-${method}-${attack}-${n}-${m}-${pcts}.p" --n ${n} --model ${model} --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --n_runs 999
