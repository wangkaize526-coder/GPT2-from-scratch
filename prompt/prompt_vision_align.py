"""
Train frame:
[IMG_TOKEN_1, IMG_TOKEN_2, ..., IMG_TOKEN_49] + <|endoftext|> {user}# {assistant} <|endoftext|>

Token ids (x, y) are pre-encoded in one pass over the jsonl.

Image tensors are NOT pre-loaded: one [3, 224, 224] float32 sample is 0.57 MiB,
so the 502k train samples would need ~287 GiB of RAM (this machine has 16 GiB).
Images are decoded lazily from images.zip inside __getitem__ instead.
"""
from datasets import load_dataset
import glob
import io
import os
import zipfile

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

IMG_SLOTS = 49      # number of image placeholder tokens in front of the text
MAX_LENGTH = 200    # longest sample is ~85 tokens, so this never truncates
BATCH_SIZE = 128
NUM_WORKERS = 0     # with spawn each worker gets its own copy of the token
                    # tensors (~1.6 GiB for train), so keep it low on a 16 GiB box

# images.zip inside the HF dataset cache. sample["image"] is a path *inside* this
# zip ("00149/001494801.jpg"), it is not a real file on disk, so it cannot be
# passed to Image.open() directly.
IMAGES_ZIP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain/"
    "snapshots/*/images.zip"))[0]

_zip = None


def _get_zip():
    """Return a per-process ZipFile handle, opened on first use.

    ZipFile objects are not picklable, and torch DataLoader workers use the
    spawn start method on macOS, so the handle must never live on the Dataset
    object (the whole dataset gets pickled to every worker). Every worker
    process ends up opening its own handle.
    """
    global _zip
    if _zip is None:
        _zip = zipfile.ZipFile(IMAGES_ZIP)
    return _zip


def _image_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145, 0.45783, 0.40821],
            std=[0.26863, 0.26130, 0.27578]
        )
    ])


def _encode_split(split, encoder):
    """Encode one split into token id tensors plus the list of image names.

    Only the jsonl is read here, no image is decoded.
    """
    x_list = []
    y_list = []
    image_paths = []
    anchor_list = []

    for sample in split:
        # Half of the samples put the "<image>" marker before the question and
        # half after it ("<image>\nWhat is this?" vs "What is this?\n<image>"),
        # so strip it wherever it is instead of only replacing "\n<image>".
        question_text = sample["conversations"][0]["value"].replace("<image>", "").strip()
        answer_text = sample["conversations"][1]["value"]

        prefix_ids = encoder.encode(f"<|endoftext|>{question_text}#", allowed_special="all")
        answer_ids = encoder.encode(f"{answer_text}<|endoftext|>", allowed_special="all")

        image_ids = [0] * IMG_SLOTS
        token_ids = image_ids + prefix_ids + answer_ids

        # index of the first answer token inside token_ids
        answer_start = len(image_ids) + len(prefix_ids)
        real_len = len(token_ids)

        pad_amount = MAX_LENGTH - real_len
        if pad_amount > 0:
            token_ids += [encoder.eot_token] * pad_amount

        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        x = token_tensor[:-1]
        y = token_tensor[1:].clone()
        # The 49 image slots and the question must not contribute to the loss.
        # y[t] is the label for predicting token t+1, so the first label we keep
        # is y[answer_start - 1], which predicts the first answer token.
        y[:answer_start - 1] = -100
        # padding part need mask
        y[real_len - 1:] = -100

        x_list.append(x)
        y_list.append(y)
        image_paths.append(sample["image"])
        anchor_list.append(real_len - 2)

    return (torch.stack(x_list), torch.stack(y_list),torch.tensor(anchor_list, dtype=torch.long), image_paths)


class VisionAlignDataset(Dataset):
    """Token ids stay in RAM (~800 MiB per split), images are decoded on demand."""

    def __init__(self, x, y, anchor_idx,image_paths, img_transform):
        self.x = x
        self.y = y
        self.anchor_idx = anchor_idx
        self.image_paths = image_paths
        self.img_transform = img_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_bytes = _get_zip().read(self.image_paths[index])
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.x[index], self.y[index], self.img_transform(pil_image),self.anchor_idx[index]


def get_dataloader_vision_align():
    train_json = "./data/pre-training/vision_alignment/train.jsonl"
    test_json = "./data/pre-training/vision_alignment/test.jsonl"

    train_split = load_dataset("json", data_files=train_json, split="train")
    test_split = load_dataset("json", data_files=test_json, split="train")

    encoder = tiktoken.get_encoding("gpt2")

    print("[vision_align] encoding train split (tokens only, no images)")
    x_train, y_train, anchor_train,paths_train = _encode_split(train_split, encoder)
    print("[vision_align] encoding test split (tokens only, no images)")
    x_test, y_test, anchor_test,paths_test = _encode_split(test_split, encoder)
    print(f"[vision_align] train x {tuple(x_train.shape)}, test x {tuple(x_test.shape)}, "
          f"images are read from {os.path.basename(IMAGES_ZIP)} at __getitem__ time")

    img_transform = _image_transform()

    dataset_train = VisionAlignDataset(x_train, y_train, anchor_train,paths_train, img_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)

    dataset_test = VisionAlignDataset(x_test, y_test, anchor_test,paths_test, img_transform)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)

    return dataloader_train, dataloader_test
