#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2

python detect.py --token_file "results/opt-${method}-10-10.p" --n 10 --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k 10 --method ${method} --n_runs 999
python detect.py --token_file "results/gpt-${method}-10-10.p" --n 10 --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k 10 --method ${method} --n_runs 999
