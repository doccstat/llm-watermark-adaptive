from os.path import exists
from os import makedirs
from datasets import load_dataset

# Load the dataset (this might take a while depending on your internet connection and the dataset size)
dataset = load_dataset("allenai/c4", "realnewslike", split="train")

# Save the dataset locally
if not exists("/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train"):
    makedirs("/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train")
    dataset.save_to_disk(
        '/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train'
    )
