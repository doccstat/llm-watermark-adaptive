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
for model_prefix in ml3 mt7; do
  if [ "$model_prefix" = "ml3" ]; then
    model="meta-llama/Meta-Llama-3-8B"
  else
    model="mistralai/Mistral-7B-v0.1"
  fi
  for watermark_key_length in 10 20 30 40 50; do
    for attack in deletion insertion substitution; do
      if [ "$attack" = "substitution" ]; then
        pcts_list=(0.0 0.1 0.2 0.3)
      elif [ "$attack" = "deletion" ]; then
        pcts_list=(1.0)
      else
        pcts_list=(1.0)
      fi
      tokens_count=$watermark_key_length
      for method in gumbel; do
        for pcts in $pcts_list; do
          echo "python 3-textgen.py --save results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts --watermark_key_length $watermark_key_length --batch_size 50 --tokens_count $tokens_count --buffer_tokens 0 --model $model --seed 1 --T 1000 --method $method --${attack} $pcts" >> 3-textgen-commands.sh
        done
      done
    done
  done
done

sbatch 3-textgen.sh
sacct -j <jobid> --format=JobID,JobName,State,ExitCode | grep textgen
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

for k_tokens_count_ratio in 0.3 0.6 1.0; do
  for model_prefix in ml3 mt7; do
    if [ "$model_prefix" = "ml3" ]; then
      model="meta-llama/Meta-Llama-3-8B"
    else
      model="mistralai/Mistral-7B-v0.1"
    fi

    for watermark_key_length in 10 20 30 40 50; do
      for attack in deletion insertion substitution; do
        if [ "$attack" = "substitution" ]; then
          pcts_list=(0.0 0.1 0.2 0.3)
        else
          pcts_list=(1.0)
        fi

        tokens_count=$watermark_key_length

        k=$(awk "BEGIN {print int($tokens_count * $k_tokens_count_ratio)}")

        for method in gumbel; do
          for pcts in "${pcts_list[@]}"; do
            mkdir -p results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts-$k-detect

            for Tindex in $(seq 0 999); do
              echo "python 4-detect.py --token_file results/${model_prefix}-${method}-${attack}-${watermark_key_length}-${tokens_count}-${pcts} --n ${watermark_key_length} --model ${model} --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --n_runs 999" >> 4-detect-commands.sh
            done
          done
        done
      done
    done
  done
done

            # "rm -rf results/$model_prefix-$method-$attack-$watermark_key_length-$tokens_count-$pcts-$k-detect"

# sbatch --dependency=afterok:<jobid> 4-detect.sh
jobid=$(sbatch --parsable 4-detect.sh)
sacct -j $jobid --format=JobID,JobName,State,ExitCode --noheader | grep detect
sacct -j $jobid --format=JobID,JobName,State,ExitCode --parsable2 | awk -F'|' '
  /detect/ {
    if ($3 == "NODE_FAIL") { node_fail++ }
    if ($3 == "PENDING") { pending++ }
    if ($3 == "COMPLETED") { completed++ }
    if ($3 == "RUNNING" ) { running++ }
  }
  END {
    print "Node fail:", node_fail
    print "Pending:", pending
    print "Completed:", completed
    print "Running:", running
  }'
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

scontrol -o show nodes | awk '{ print $1, $4, $10, $18, $25, $26, $27}'
scontrol -d show job 389854 | grep Reason
