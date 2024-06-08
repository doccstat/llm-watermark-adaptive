#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2
attack=$3
pcts=$4

python detect.py --token_file "results/opt-${method}-${attack}-30-30-${pcts}.p" --n 30 --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k 30 --method ${method} --n_runs 999
python detect.py --token_file "results/gpt-${method}-${attack}-30-30-${pcts}.p" --n 30 --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k 30 --method ${method} --n_runs 999
