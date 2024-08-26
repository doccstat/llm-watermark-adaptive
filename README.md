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
rm -f 3-textgen-commands.sh
for watermark_key_length in 20 50 100 500 1000; do
  if [ $watermark_key_length -le 100 ]; then
    tokens_count=$watermark_key_length
  else
    tokens_count=100
  fi

  for method in gumbel; do
    for attack in deletion insertion substitution; do
      for pcts in 0.0 0.05 0.1 0.2 0.3; do
        for model_prefix in ml3 mt7; do
          if [ "$model_prefix" = "ml3" ]; then
            model="meta-llama/Meta-Llama-3-8B"
          else
            model="mistralai/Mistral-7B-v0.1"
          fi

          echo "python 3-textgen.py --save results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 50 --tokens_count $tokens_count --buffer_tokens 0 --model $model --seed 1 --T 100 --method $method --${attack} $pcts --candidate_prompt_max 10 --gpt_prompt_key ''" >> 3-textgen-commands.sh
        done
      done
    done
  done
done

sbatch 3-textgen.sh
# sacct -j <jobid> --format=JobID,JobName,State,ExitCode
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
for watermark_key_length in 20 50 80 100 500 1000; do
  if [ $watermark_key_length -le 100 ]; then
    tokens_count=$watermark_key_length
  else
    tokens_count=100
  fi

  for method in gumbel; do
    for attack in deletion insertion substitution; do
      for pcts in 0.0 0.05 0.1 0.2 0.3; do
        for model_prefix in ml3 mt7; do
          if [ "$model_prefix" = "ml3" ]; then
            model="meta-llama/Meta-Llama-3-8B"
          else
            model="mistralai/Mistral-7B-v0.1"
          fi

          rm -rf results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p-detect
          mkdir -p results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p-detect
          for Tindex in $(seq 0 99); do
            echo "python 4-detect.py --token_file results/${model_prefix}-${method}-${attack}-${watermark_key_length}-${tokens_count}-${pcts}.p --n ${watermark_key_length} --model ${model} --seed 1 --Tindex ${Tindex} --k ${tokens_count} --method ${method} --n_runs 999" >> 4-detect-commands.sh
          done
        done
      done
    done
  done
done

# sbatch --dependency=afterok:<jobid> 4-detect.sh
sbatch 4-detect.sh
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
# tokenizer.decode([374,279,9647,315,279,88931,16483,13,8595,433,374,430,584,527,6982,912,3585,315,70000,1990], skip_special_tokens=True)
# ' is the opinion of the foregoing writers. Why it is that we are shown no points of resemblance between'
```
