#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2

python detect.py --token_file "results/opt-${method}-20-20.p" --n 20 --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k 20 --method ${method} --n_runs 499
python detect.py --token_file "results/gpt-${method}-20-20.p" --n 20 --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k 20 --method ${method} --n_runs 499
