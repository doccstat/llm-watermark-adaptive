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

          echo "python 3-textgen.py --save results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts.p --watermark_key_length $watermark_key_length --batch_size 150 --tokens_count $tokens_count --buffer_tokens 0 --model $model --seed 1 --T 300 --method $method --${attack} $pcts --candidate_prompt_max 20" >> 3-textgen-commands.sh
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
          if [ "$model_prefix" = "opt" ]; then
              model="facebook/opt-1.3b"
          elif [ "$model_prefix" = "gpt" ]; then
              model="openai-community/gpt2"
          elif [ "$model_prefix" = "ml3" ]; then
              model="meta-llama/Meta-Llama-3-8B"
          elif [ "$model_prefix" = "mt7" ]; then
              model="mistralai/Mistral-7B-v0.1"
          else
              echo "Invalid model prefix"
              exit 1
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
