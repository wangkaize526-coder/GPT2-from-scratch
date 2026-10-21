from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import TensorDataset, DataLoader
'''
Train frame:
49*img_token <|endoftext|> text <|endoftext|>
'''

def build_wiki_dataloader():
    x_train_list= []
    y_train_list= []
    x_val_list= []
    y_val_list= []
    wiki_data_train= load_dataset("json", data_files= "./data/pre-training/wiki/train.json")["train"]
    wiki_data_val= load_dataset("json", data_files= "./data/pre-training/wiki/test.json")["train"]

    encoder = tiktoken.get_encoding("gpt2")
    SLOTS= 49
    max_len= 251
    for sample in wiki_data_train:
            body = encoder.encode(f"<|endoftext|>{sample['text']}<|endoftext|>", allowed_special="all")
            token_ids = [0]*SLOTS + body       # <-- slots FIRST
            real_len = len(token_ids)          # real_len now INCLUDES the 49 slots
            if real_len > max_len:             # guard (needed, see below)
                continue
            token_ids += [encoder.eot_token] * (max_len - real_len)
    
            x = torch.tensor(token_ids[:-1], dtype=torch.long)
            y = torch.tensor(token_ids[1:],  dtype=torch.long).clone()
            y[:SLOTS] = -100
            y[real_len-1:] = -100
            x_train_list.append(x)
            y_train_list.append(y)

    for sample in wiki_data_val:
            body = encoder.encode(f"<|endoftext|>{sample['text']}<|endoftext|>", allowed_special="all")
            token_ids = [0]*SLOTS + body       # <-- slots FIRST
            real_len = len(token_ids)          # real_len now INCLUDES the 49 slots
            if real_len > max_len:             # guard (needed, see below)
                continue
            token_ids += [encoder.eot_token] * (max_len - real_len)
    
            x = torch.tensor(token_ids[:-1], dtype=torch.long)
            y = torch.tensor(token_ids[1:],  dtype=torch.long).clone()
            y[:SLOTS] = -100
            y[real_len-1:] = -100
            x_val_list.append(x)
            y_val_list.append(y)

    x_all_train = torch.stack(x_train_list, dim=0)
    y_all_train = torch.stack(y_train_list, dim=0)
    x_all_test = torch.stack(x_val_list, dim=0)
    y_all_test = torch.stack(y_val_list, dim=0)

    dataset_train = TensorDataset(x_all_train, y_all_train)
    dataset_test = TensorDataset(x_all_test, y_all_test)

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_test

