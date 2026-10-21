"""Does the trained model actually use the image?

Three passes over the same align test batches, identical in every way except
what is fed into the 49 vision slots:
    normal   - the real image
    shuffled - the SAME batch's images, permuted (so the caption is now wrong
               for the picture; any model that reads the image must get worse)
    zero     - a blank image

If shuffled/zero score the same as normal, the vision encoder's output is not
reaching the answer tokens at all - the align loss was learned from the caption
prior alone and no amount of extra alignment training will fix it.

Run from the project root:  CUDA_VISIBLE_DEVICES=2 python logs/probe_vision.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

import prompt.prompt_vision_align as pva
from model.gpt2 import GPT2

N_BATCHES = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2().to(device)
model.load_state_dict(torch.load("check_point/gpt2_model_best.pth", map_location=device))
model.eval()
print(f"[probe] loaded best checkpoint onto {device}", flush=True)

_, dl_test = pva.get_dataloader_vision_align()
batches = []
for i, item in enumerate(dl_test):
    if i >= N_BATCHES:
        break
    batches.append(item)
print(f"[probe] {len(batches)} test batches = "
      f"{sum(b[0].size(0) for b in batches)} samples", flush=True)


@torch.no_grad()
def align_ce(mode):
    total_nll = total_tok = 0
    for x, y, img ,anchor in batches:
        x, y, img = x.to(device), y.to(device), img.to(device)
        if mode == "shuffled":
            img = img[torch.randperm(img.size(0), device=img.device)]
        elif mode == "zero":
            img = torch.zeros_like(img)
        logits = model(x, img)
        total_nll += nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            ignore_index=-100, reduction="sum").item()
        total_tok += (y != -100).sum().item()
    return total_nll / max(total_tok, 1)


results = {m: align_ce(m) for m in ("normal", "shuffled", "zero")}

print()
print(f"[probe] align cross-entropy over {sum(b[0].size(0) for b in batches)} samples")
for m, ce in results.items():
    print(f"    {m:9s} ce {ce:.4f}   ppl {math.exp(ce):8.2f}")

delta_shuf = results["shuffled"] - results["normal"]
delta_zero = results["zero"] - results["normal"]
print()
print(f"[probe] shuffled - normal = {delta_shuf:+.4f} nats")
print(f"[probe] zero     - normal = {delta_zero:+.4f} nats")
print()
if delta_shuf < 0.05:
    print("[probe] VERDICT: the image barely matters. Scrambling every image in the"
          " batch changes the loss by less than 0.05 nats - the model is answering"
          " from the caption prior, not from what it sees.")
else:
    print("[probe] VERDICT: the image is being read - breaking it measurably hurts.")
