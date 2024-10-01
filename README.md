# llm-watermark-cpd

## Prerequisites

<details closed>
<summary>Python environments</summary>

-   Cython==3.0.10
-   datasets==2.19.1
-   huggingface_hub==0.23.0
-   nltk==3.8.1
-   numpy==1.26.4
-   sacremoses==0.0.53
-   scipy==1.13.0
-   sentencepiece==0.2.0
-   tokenizers==0.19.1
-   torch==2.3.0.post100
-   torchaudio==2.3.0
-   torchvision==0.18.0
-   tqdm==4.66.4
-   transformers==4.40.2

</details>

### Set up environments

#### Python

> [!NOTE]
> Refer to https://pytorch.org for PyTorch installation on other platforms

```shell
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
```

#### R

> [!NOTE]
> R is used for change point detection. Refer to https://www.r-project.org for
> installation instructions.

```r
install.packages(c("doParallel", "reshape2", "ggplot2", "fossil"))
```

## Instruction

To reproduce the results, follow the instructions below or use the attached
results directly using `Rscript 5-analyze.R 1 3200`.

### Set up pyx

```shell
python 1-setup.py build_ext --inplace
```

#### Expected running time

Less than 1 minute on a single core CPU machine.

#### Expected memory usage

Less than 1 GB.

### (Optional) Download the pre-trained language model

```shell
sbatch 2-download.sh
```

### Generate watermarked tokens

```shell
rm -f 3-textgen-instruct-commands.sh
for watermark_key_length in 10 20 30; do
  tokens_count=$watermark_key_length

  for method in gumbel; do
    for attack in deletion insertion substitution; do
      for pcts in 0.0 0.1 0.2 0.3; do
        for model_prefix in ml3; do
          if [ "$model_prefix" = "ml3" ]; then
            model="meta-llama/Meta-Llama-3-8B"
          else
            model="mistralai/Mistral-7B-v0.1"
          fi

          echo "python 3-textgen-instruct.py --save results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 100 --tokens_count $tokens_count --buffer_tokens 0 --model $model --seed 1 --T 500 --method $method --${attack} $pcts --candidate_prompt_max 0 --gpt_prompt_key ''" >> 3-textgen-instruct-commands.sh
        done
      done
    done
  done
done

sbatch 3-textgen-instruct.sh
sacct -j 412046 --format=JobID,JobName,State,ExitCode | grep textgen

rm -f 3-textgen-vocab-commands.sh
for watermark_key_length in 10; do
  tokens_count=$watermark_key_length

  for method in gumbel; do
    for attack in deletion; do
      for pcts in 0.0; do
        for model_prefix in ml3 mt7; do
          if [ "$model_prefix" = "ml3" ]; then
            model="meta-llama/Meta-Llama-3-8B"
          else
            model="mistralai/Mistral-7B-v0.1"
          fi

          echo "python 3-textgen-vocab.py --save results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 100 --tokens_count $tokens_count --buffer_tokens 0 --model $model --seed 1 --T 100 --method $method --${attack} $pcts --gpt_prompt_key ''" >> 3-textgen-vocab-commands.sh
        done
      done
    done
  done
done

sbatch 3-textgen-vocab.sh
sacct -j 406611 --format=JobID,JobName,State,ExitCode | grep textgen
```

#### Expected running time

Less than 5 hours on 1 compute node with 1 NVIDIA A30 GPU and 128 CPU cores.

#### Expected memory usage

Less than 128 GB.

> [!NOTE]
> Variables starting with `_` are not being used and safe to remove in
> future development.

### Calculate p-values for texts

```shell
rm -f 4-detect-commands.sh
for watermark_key_length in 10 20 30; do
  tokens_count=$watermark_key_length

  for method in gumbel; do
    for attack in deletion insertion substitution; do
      for pcts in 0.1 0.2 0.3; do
        for model_prefix in ml3 mt7; do
          if [ "$model_prefix" = "ml3" ]; then
            model="meta-llama/Meta-Llama-3-8B"
          else
            model="mistralai/Mistral-7B-v0.1"
          fi

          rm -rf results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p-detect
          mkdir -p results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p-detect
          for Tindex in $(seq 0 99); do
            echo "python 4-detect.py --token_file results/${model_prefix}-${method}-${attack}-${watermark_key_length}-${tokens_count}-${pcts}.p --n ${watermark_key_length} --model ${model} --seed 1 --Tindex ${Tindex} --k 5 --method ${method} --n_runs 99" >> 4-detect-commands.sh
          done
        done
      done
    done
  done
done

# sbatch --dependency=afterok:409989 4-detect.sh
sbatch 4-detect.sh
sacct -j 412608 --format=JobID,JobName,State,ExitCode | grep detect | grep COMPLETED | wc -l
```

#### Collect results

```shell
tar -czvf results-20-20-10.tar.gz results
tar -xzvf results-20-20-10.tar.gz
```

#### Expected running time

Less than 24 hours on 8 compute nodes with no GPU and 28 CPU cores each.

#### Expected memory usage

Less than 10 GB per compute node.

### Analysis

```shell
parallel -j 8 --progress Rscript 5-analyze.R {1} {2} ::: $(seq 1 400 2801) ::: $(seq 400 400 3200)
Rscript 5-analyze.R 1 3200
```

#### Expected running time

Less than 12 hours on 8 compute nodes with no GPU and 28 CPU cores each.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("/scratch/user/anthony.li/models/" + "meta-llama/Meta-Llama-3-8B" + "/tokenizer")
# icl prompt
tokenizer.decode([128000,31437,43874,656,279,5370,12283,499,3077,14407,3118,304,872,4477,30,83017,701,4320,11,1524,422,279,12283,2873,311,617,912,2867,3585,315,70000,1210,128000,128000,128000], skip_special_tokens=True)
# '"What similarities do the various authors you\'ve discussed present in their writing? Explain your answer, even if the authors seem to have no clear points of resemblance."'
# true prompt
tokenizer.decode([33235,430,279,9578,374,264,26436,574,14592,505,279,12917,315,14154,3980,79454,11,323,813,16801,304,279,14209,315,279,3276,575,2601,574,4762,28160,555,279,9815,902,279,14154,18088,23933,11467,6688,315,872,64876,13,1115,11,520,3325,11], skip_special_tokens=True)
# ' doctrine that the earth is a sphere was derived from the teaching of ancient geographers, and his belief in the existence of the antipodes was probably influenced by the accounts which the ancient Irish voyagers gave of their journeys. This, at least,'
# text
tokenizer.decode([374,279,9647,315,279,88931,16483,13,8595,433,374,430,584,527,6982,912,3585,315,70000,1990], skip_special_tokens=True)
# ' is the opinion of the foregoing writers. Why it is that we are shown no points of resemblance between'

tokenizer = AutoTokenizer.from_pretrained("/scratch/user/anthony.li/models/" + "mistralai/Mistral-7B-v0.1" + "/tokenizer")
tokenizer.decode([1,315,837,1404,298,1464,298,712,529,264,18958,438,1611,354,7812,3322,2449,456,879,28723,1824,7108,304,9804,511,315,927,298,10130,625,264,18958,712,13072,28804])
# '<s> I am going to try to roast a pig at home for Thanksgiving this year. What equipment and techniques do I need to successfully get a pig roasted?'
tokenizer.decode([1418,4349,28705,28750,28734,28725,28705,28750,28734,28734])
# 'On November 20, 200'
```

scontrol -o show nodes | awk '{ print $1, $4, $10, $18, $25, $26, $27}'
scontrol -d show job 389854 | grep Reason

```r
# >>> import torch
# r, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("/scratch/user/anthony.li/models/" + "openai-community/gpt2" + "/tokenizer")
# >>> from transformers import AutoTokenizer, AutoModelForCausalLM
# >>> tokenizer = AutoTokenizer.from_pretrained("/scratch/user/anthony.li/models/" + "openai-community/gpt2" + "/tokenizer")
# >>> tokenizer.decode([34442, 37250, 31524, 17241, 18161, 23068, 26933, 2061, 22222, 2131, 1980, 8888, 20954, 49366, 43984, 21474, 2150, 26792, 8135, 39738])
'advertising [\'Basically],"."[ (£([WhatinburghciallyircTodayésworldlyBangchtungingessk neuroscience'
# >>> tokenizer.decode([74,710,21474,2150,82,4914,928,494,12340,11769])
'knechtungskaologie"), Wales'
# >>> tokenizer.decode([7743,286,6156,4903,34063,11,290,465,4901,287,262,6224,286,262,32867,4147,373,2192,12824,416,262,5504,543,262,6156,8685,23291,10321,2921,286,511,35724,13,770,11,379,1551,11,318,262,4459,286,371,3087,3900,5855,42,1980,31753,274])
' teaching of ancient geographers, and his belief in the existence of the antipodes was probably influenced by the accounts which the ancient Irish voyagers gave of their journeys. This, at least, is the opinion of Rettberg ("Kirchenges'
# >>> tokenizer.decode([679,338,257,845,922,393,1049,3985,13,198])
" He's a very good or great coach.\n"
# >>> tokenizer.decode([14536,422,3050,12,1314,11,326,339,373,257,1310,3491,19554,694,13,198,1537,1754,318,7725,326,530,1110,339,460,2666,257,10655,588,27059,259,393,5030,13,887,329,783,11,465,3061,318,284,5879,644,339,1541,5804,318,2081,25])
' Lions from 2010-15, that he was a little starstruck.\nButler is hoping that one day he can leave a legacy like Boldin or Johnson. But for now, his goal is to prove what he already believes is true:'
# >>> tokenizer.decode([29582,39737,50256,1639,257,845,922,18680,30906,3985,32904,49537,50256,464,393,3064,8842,3985,42597,44962])
' Pastebinquished<|endoftext|>You a very good mightyrawdownloadcloneembedreportprint coach miracles Heisman<|endoftext|>The or100 fantasy coachSHAREGOP'
```
