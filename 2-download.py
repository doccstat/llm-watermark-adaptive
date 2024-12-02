from transformers import AutoTokenizer, AutoModelForCausalLM

from os.path import exists
from os import makedirs
from datasets import load_dataset

# Load the dataset (this might take a while depending on the internet
# connection and the dataset size)
dataset = load_dataset("allenai/c4", "realnewslike", split="train")
dataset_path = "/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train"

# Save the dataset locally
if not exists(dataset_path):
    makedirs(dataset_path)
    dataset.save_to_disk(dataset_path)

model_path = "/scratch/user/anthony.li/models/"

for model_name in [
    "meta-llama/Meta-Llama-3-8B",
    "openai-community/gpt2",
    "facebook/opt-1.3b",
    "mistralai/Mistral-7B-v0.1"
]:
    if not exists(model_path + model_name):
        makedirs(model_path + model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path + model_name + "/tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(model_path + model_name + "/model")
