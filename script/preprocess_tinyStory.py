from datasets import load_dataset
import pandas as pd
'''
    dataset = load_dataset("roneneldan/TinyStories")
    # store the datast to jsonl file in data folder
    dataset['train'].to_json("data/tinyStories_train.jsonl")
    dataset['validation'].to_json("data/tinyStories_validation.jsonl")
'''
train_df = pd.read_json("data/tinyStories_train.jsonl", lines=True)
train_df.info()

