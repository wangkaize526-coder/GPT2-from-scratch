"""Is the vision representation collapsed, or is the LM just ignoring it?

The probe_vision.py result (scrambling images costs only 0.30 nats) has two very
different possible causes:

  (a) the ViT + projector map every image to nearly the SAME 49x512 tensor, so
      there is nothing for the LM to read, or
  (b) the representation does differ per image, but the LM does not use it.

This measures the vision tower output directly. For a batch of 32 different
photos it reports the mean pairwise cosine similarity between the 49x512
tensors. ~1.0 means collapse (a); near 0 means healthy and the problem is (b).

It also compares the vision tensor against the text embedding table and against
no_image_emb, to see what the LM was being handed alongside real tokens.

Run from the project root:  CUDA_VISIBLE_DEVICES=2 python logs/probe_repr.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

import prompt.prompt_vision_align as pva
from model.gpt2 import GPT2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2().to(device)
model.load_state_dict(torch.load("check_point/gpt2_model_best.pth", map_location=device))
model.eval()

dl_train, _ = pva.get_dataloader_vision_align()
x, y,img, anchor = next(iter(dl_train))
img = img.to(device)
print(f"[repr] batch of {img.size(0)} distinct images", flush=True)

with torch.no_grad():
    raw = model.vision_encoder(img)              # (B, 49, 512) straight from the ViT
    emb = model.projector(raw)                   # (B, 49, 512) what forward() installs
    noimg = model.no_image_emb.expand(img.size(0), -1, -1)

    # flatten each image's 49x512 tensor to one vector, then compare directions
    def pairwise_cos(t):
        v = F.normalize(t.reshape(t.size(0), -1), dim=-1)
        sim = v @ v.t()
        n = sim.size(0)
        return (sim.sum() - sim.trace()) / (n * (n - 1))

    print(f"[repr] mean pairwise cosine, ViT output (before projector): {pairwise_cos(raw):.4f}")
    print(f"[repr] mean pairwise cosine, after projector            : {pairwise_cos(emb):.4f}")
    print(f"[repr] mean pairwise cosine, no_image_emb vs itself     : "
          f"{pairwise_cos(noimg):.4f}")
    print()
    print(f"[repr] per-dim std across the 32 images, after projector: "
          f"{emb.std(dim=0).mean().item():.4f}")
    print(f"[repr] per-dim std across the 32 images, ViT output     : "
          f"{raw.std(dim=0).mean().item():.4f}")
    print()
    # what scale is the LM used to? the text embedding table it was pretrained with
    print(f"[repr] token_emb per-row norm   : {model.token_emb.weight.norm(dim=-1).mean().item():.4f}")
    print(f"[repr] vision emb per-row norm  : {emb.norm(dim=-1).mean().item():.4f}")
    print(f"[repr] no_image_emb per-row norm: {noimg.norm(dim=-1).mean().item():.4f}")
    print()
    # does the projector just collapse toward no_image_emb?
    print(f"[repr] cos(vision emb, no_image_emb) : "
          f"{F.cosine_similarity(emb.reshape(1, -1), noimg.reshape(1, -1)).item():.4f}")
