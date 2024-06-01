#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2

python detect.py --token_file "results/opt-${method}-40-40.p" --n 40 --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k 40 --method ${method}
python detect.py --token_file "results/gpt-${method}-40-40.p" --n 40 --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k 40 --method ${method}
