#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2
attack=$3
pcts=$4

n=20
m=20
k=10

python detect.py --token_file "results/opt-${method}-${attack}-${n}-${m}-${pcts}.p" --n ${n} --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --n_runs 999
python detect.py --token_file "results/gpt-${method}-${attack}-${n}-${m}-${pcts}.p" --n ${n} --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --n_runs 999
