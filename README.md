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

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
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
bash 3-textgen-helper.sh
sbatch 3-textgen.sh
sacct -j $jobid --format=JobID,JobName,State,ExitCode | grep textgen
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
bash 4-detect-helper.sh
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
