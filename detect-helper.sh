#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2
attack=$3
pcts=$4

python detect.py --token_file "results/opt-${method}-${attack}-10-10-${pcts}.p" --n 10 --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k 10 --method ${method} --n_runs 999
python detect.py --token_file "results/gpt-${method}-${attack}-10-10-${pcts}.p" --n 10 --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k 10 --method ${method} --n_runs 999
