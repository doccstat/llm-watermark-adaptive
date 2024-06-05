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

### Generate watermarked tokens

```shell
mkdir -p results
mkdir -p log

export PYTHONPATH=".":$PYTHONPATH

for method in gumbel; do
  for pcts in 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    python textgen.py --save results/opt-$method-deletion-10-10-$pcts.p --n 10 --batch_size 50 --m 10 --model facebook/opt-1.3b --seed 1 --T 1000 --method $method --deletion $pcts
    python textgen.py --save results/opt-$method-insertion-10-10-$pcts.p --n 10 --batch_size 50 --m 10 --model facebook/opt-1.3b --seed 1 --T 1000 --method $method --insertion $pcts
    python textgen.py --save results/opt-$method-substitution-10-10-$pcts.p --n 10 --batch_size 50 --m 10 --model facebook/opt-1.3b --seed 1 --T 1000 --method $method --substitution $pcts
  done
done

for method in gumbel; do
  for pcts in 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    python textgen.py --save results/gpt-$method-deletion-10-10-$pcts.p --n 10 --batch_size 50 --m 10 --model openai-community/gpt2 --seed 1 --T 1000 --method $method --deletion $pcts
    python textgen.py --save results/gpt-$method-insertion-10-10-$pcts.p --n 10 --batch_size 50 --m 10 --model openai-community/gpt2 --seed 1 --T 1000 --method $method --insertion $pcts
    python textgen.py --save results/gpt-$method-substitution-10-10-$pcts.p --n 10 --batch_size 50 --m 10 --model openai-community/gpt2 --seed 1 --T 1000 --method $method --substitution $pcts
  done
done
```

#### Expected running time

Less than 2 hours on 1 compute node with 1 NVIDIA A30 GPU and 128 CPU cores.

#### Expected memory usage

Less than 128 GB.

### Calculate p-values for texts

```shell
for method in gumbel; do
  for model in opt gpt; do
    for pcts in 0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
      for attack in deletion insertion substitution; do
        mkdir -p results/$model-$method-$attack-10-10-$pcts.p-detect
      done
    done
  done
done

chmod +x ./detect.sh
parallel -j 10 --progress ./detect.sh {1} {2} ::: gumbel ::: $(seq 1 10)
```

#### Expected running time

Less than 24 hours on 8 compute nodes with no GPU and 28 CPU cores each.

#### Expected memory usage

Less than 10 GB per compute node.

### Change point analysis

```shell
parallel -j 8 --progress Rscript analyze.R {1} {2} ::: $(seq 1 400 2801) ::: $(seq 400 400 3200)
Rscript analyze.R 1 3200
```

#### Expected running time

Less than 12 hours on 8 compute nodes with no GPU and 28 CPU cores each.

##### Running time test

The following command should run in less than 10 minutes on 1 compute node
with no GPU and 28 CPU cores.

```shell
Rscript analyze.R 1 5
```

#### Expected memory usage

Less than 10 GB per compute node.
