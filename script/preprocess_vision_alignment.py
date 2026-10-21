from datasets import load_dataset
import os
import glob
import json

# 创建文件夹
save_root = "./data/pre-training/vision_alignment"
os.makedirs(save_root, exist_ok=True)

# Load the locally cached JSON file directly.
# The repo also ships blip_laion_cc_sbu_558k_meta.json (id/image/blip_caption/url),
# so the default data_files glob merges two files with different columns and fails
# with CastError. Point at the single file we need, via its cached absolute path
# (a relative data_files path requires Hub access to resolve, which fails offline).
LLAVA_JSON = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain/"
    "snapshots/*/blip_laion_cc_sbu_558k.json"))[0]

ds = load_dataset("json", data_files=LLAVA_JSON, split="train")
ds_split = ds.train_test_split(test_size=0.1, seed=42)
train_ds = ds_split["train"]
test_ds = ds_split["test"]

# 输出 train.jsonl
train_path = os.path.join(save_root, "train.jsonl")
with open(train_path, "w", encoding="utf-8") as f:
    for sample in train_ds:
        # jsonl：不要indent，一行一条
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")

# 输出 test.jsonl
test_path = os.path.join(save_root, "test.jsonl")
with open(test_path, "w", encoding="utf-8") as f:
    for sample in test_ds:
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")

print(f"Done! Train: {len(train_ds)}, Test: {len(test_ds)}")
print(f"Saved to {save_root}")
