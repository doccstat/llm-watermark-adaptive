# A Likelihood Based Approach for Watermark Detection

Implementation of the methods described in "A Likelihood Based Approach for Watermark Detection" by [Xingchi Li](https://xingchi.li), [Guanxun Li](https://guanxun.li), [Xianyang Zhang](https://zhangxiany-tamu.github.io).

[![OpenReview](https://img.shields.io/badge/OpenReview-A%20Likelihood%20Based%20Approach%20for%20Watermark%20Detection-8c1b13.svg)](https://openreview.net/forum?id=S2QoDt4bw4)

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

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
```

## Instructions

All experiments are conducted using Slurm workload manager. Expected running
time and memory usage are provided in the corresponding sbatch scripts.

> [!IMPORTANT]
> Please modify the paths, Slurm mail options and adjust the GPU resources in
> the sbatch scripts before running the experiments.

```shell
# Setup pyx.
sbatch 1-setup.sh

# Download models to local.
sbatch 2-download.sh

# Text generation.
bash 3-textgen-helper.sh
sbatch 3-textgen.sh

# Watermark detection.
bash 4-detect-helper.sh
sbatch 4-detect.sh

# Result analysis and ploting.
Rscript 5-analyze.R
```

## Citation

```bibtex
@inproceedings{
  anonymous2025a,
  title={A Likelihood Based Approach for Watermark Detection},
  author={Anonymous},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
  year={2025},
  url={https://openreview.net/forum?id=S2QoDt4bw4}
}
```
