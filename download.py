from datasets import load_dataset

# Load the dataset (this might take a while depending on your internet connection and the dataset size)
dataset = load_dataset("allenai/c4", "realnewslike", split="train")

# Save the dataset locally
dataset.save_to_disk('/scratch/user/anthony.li/datasets/c4_realnewslike_train/')
