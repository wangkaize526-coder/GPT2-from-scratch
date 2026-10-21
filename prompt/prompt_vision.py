"""
Train frame:
[IMG_TOKEN_1, IMG_TOKEN_2, ..., IMG_TOKEN_49] + <|endoftext|> Question text # Answer text <|endoftext|>

Ids are encoded with allowed_special="all", so each <|endoftext|> marker is
really the single special token 50256, not 7 literal characters.

"""
import tiktoken
from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.vision_encoder import MyVisionEncoder

def build_dataloader_for_VLM():

    vision_data= load_dataset("json", data_files="./data/pet/cat_dog.json")["train"]
    split_data = vision_data.train_test_split(test_size=0.1, seed=42)
    encoder= tiktoken.get_encoding("gpt2")

    train_ds= split_data["train"]
    test_ds= split_data["test"]
    max_length= 90

    x_list_train=[]
    y_list_train=[]
    img_list_train= [] # to store the image tensor 

    x_list_test=[]
    y_list_test=[]
    img_list_test= [] # to store the image tensor 

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),          # resize to shape 224×224
        transforms.ToTensor(),                  
        transforms.Normalize(
            mean=[0.48145, 0.45783, 0.40821],
            std=[0.26863, 0.26130, 0.27578]
        )
    ])
    img_place_holder= 0

    # build the train dataset
    for item in train_ds:
        # get the image token
        pil_image = Image.open(item["image_path"]).convert("RGB")
        img_tensor = img_transform(pil_image) # [ 3, 224, 224]
        img_list_train.append(img_tensor)

        # Encode question and answer SEPARATELY so we know exactly where the
        # answer starts. Searching for the "#" token does NOT work: "#" is a
        # normal BPE token, so it merges with the preceding "?" / "." into one
        # token (~16% of samples have no standalone "#" at all).
        prefix_ids = encoder.encode(f"<|endoftext|>{item['question']}#", allowed_special="all")
        answer_ids = encoder.encode(f"{item['answer']}<|endoftext|>", allowed_special="all")

        padding_img_ids= [img_place_holder] * 49
        token_ids = padding_img_ids + prefix_ids + answer_ids
        # index of the first answer token inside token_ids
        answer_start = len(padding_img_ids) + len(prefix_ids)
        real_len= len(token_ids)
        # padding
        pad_amount= max_length - real_len
        if pad_amount > 0:
            token_ids += [encoder.eot_token] * pad_amount

        token_tensor= torch.tensor(token_ids, dtype=torch.long)
        x= token_tensor[:-1]
        y= token_tensor[1:].clone()
        # The 49 image slots and the question must not contribute to the loss.
        # y[t] is the label for predicting token t+1, so the first label we keep
        # is y[answer_start - 1], which predicts the first answer token.
        y[:answer_start - 1] = -100
        # padding part need mask
        y[real_len-1:] = -100
        x_list_train.append(x)
        y_list_train.append(y)

    for item in test_ds:
        pil_image = Image.open(item["image_path"]).convert("RGB")
        img_tensor = img_transform(pil_image) # [ 3, 224, 224]
        img_list_test.append(img_tensor)
        
        prefix_ids = encoder.encode(f"<|endoftext|>{item['question']}#", allowed_special="all")
        answer_ids = encoder.encode(f"{item['answer']}<|endoftext|>", allowed_special="all")

        padding_img_ids= [img_place_holder] * 49
        token_ids = padding_img_ids + prefix_ids + answer_ids
        answer_start = len(padding_img_ids) + len(prefix_ids)
        real_len= len(token_ids)
        pad_amount= max_length - real_len
        if pad_amount > 0:
            token_ids += [encoder.eot_token] * pad_amount

        token_tensor= torch.tensor(token_ids, dtype=torch.long)
        x= token_tensor[:-1]
        y= token_tensor[1:].clone()
        y[:answer_start - 1] = -100
        y[real_len-1:] = -100
        x_list_test.append(x)
        y_list_test.append(y)

    x_all_train = torch.stack(x_list_train, dim=0)
    y_all_train = torch.stack(y_list_train, dim=0)
    img_all_train = torch.stack(img_list_train, dim=0) # [N,3,224,224]
    x_all_test = torch.stack(x_list_test, dim=0)
    y_all_test = torch.stack(y_list_test, dim=0)
    img_all_test = torch.stack(img_list_test, dim=0) # [N,3,224,224]

    dataset_train = TensorDataset(x_all_train, y_all_train, img_all_train)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

    dataset_test = TensorDataset(x_all_test, y_all_test, img_all_test)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_test
