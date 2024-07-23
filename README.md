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
results directly using `Rscript analyze.R 1 3200`.

### Set up pyx

```shell
python setup.py build_ext --inplace
```

#### Expected running time

Less than 1 minute on a single core CPU machine.

#### Expected memory usage

Less than 1 GB.

### (Optional) Download the pre-trained language model

```shell
sbatch download.sh
```

### Generate watermarked tokens

```shell
sbatch textgen.sh
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
rm -f detect-commands.sh
for method in gumbel; do
  for attack in deletion insertion substitution; do
    for pcts in 0.0 0.05 0.1 0.2 0.3; do
      for llm in gpt opt; do
        rm -rf results/$llm-$method-$attack-20-20-$pcts.p-detect
        mkdir -p results/$llm-$method-$attack-20-20-$pcts.p-detect
      done
      for Tindex in $(seq 0 399); do
        echo "bash ./detect-helper.sh $method $Tindex $attack $pcts" >> detect-commands.sh
      done
    done
  done
done

previous_job_id=""
for offset in {0..5000..1000}; do
    if [[ -z $previous_job_id ]]; then
        job_id=$(sbatch detect.sh -- --offset=$offset | awk '{print $4}')
        echo "Submitted first job with ID $job_id"
    else
        job_id=$(sbatch --dependency=afterany:$previous_job_id detect.sh -- --offset=$offset | awk '{print $4}')
        echo "Submitted job with ID $job_id, dependent on job $previous_job_id"
    fi
    previous_job_id=$job_id
done
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
parallel -j 8 --progress Rscript analyze.R {1} {2} ::: $(seq 1 400 2801) ::: $(seq 400 400 3200)
Rscript analyze.R 1 3200
```

#### Expected running time

Less than 12 hours on 8 compute nodes with no GPU and 28 CPU cores each.
