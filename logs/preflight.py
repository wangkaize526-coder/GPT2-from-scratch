"""Pre-flight for Main.py: builds both dataloaders, runs a few real steps and the
eval path, and writes NOTHING to check_point/ so the real run still starts clean.

Run from the project root:  PYTHONPATH=. python logs/preflight.py
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

import prompt.prompt_wiki
import prompt.prompt_vision_align
from model.gpt2 import GPT2
from Train_and_Eval import eval_stream, make_arch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[preflight] device={device} "
      f"name={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'}", flush=True)

t0 = time.time()
dl_wiki, dl_wiki_test = prompt.prompt_wiki.build_wiki_dataloader()
print(f"[preflight] wiki loaders: train {len(dl_wiki)} batches, "
      f"test {len(dl_wiki_test)} batches  ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
dl_align, dl_align_test = prompt.prompt_vision_align.get_dataloader_vision_align()
print(f"[preflight] align loaders: train {len(dl_align)} batches, "
      f"test {len(dl_align_test)} batches  ({time.time()-t0:.1f}s)", flush=True)

# the K the real run will use, without running it
epochs_num = 5
K = max(1, round(epochs_num * len(dl_wiki) / len(dl_align)))
print(f"[preflight] len(wiki_train)={len(dl_wiki)} len(align_train)={len(dl_align)} -> K={K}", flush=True)

model = GPT2().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[preflight] GPT2 params {n_params/1e6:.1f}M  arch={make_arch(model)}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# ---- a few real text steps -------------------------------------------------
model.train()
x, y = next(iter(dl_wiki))
x, y = x.to(device), y.to(device)
print(f"[preflight] wiki batch x{tuple(x.shape)} y{tuple(y.shape)} "
      f"labels_kept={(y != -100).sum().item()}", flush=True)
for i in range(2):
    loss = criterion(model(x).view(-1, 50257), y.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"[preflight] wiki step {i+1} text_ce {loss.item():.4f}", flush=True)

# ---- a few real align steps (decodes images from images.zip) ---------------
it = iter(dl_align)
xa, ya, ia = next(it)
xa, ya, ia = xa.to(device), ya.to(device), ia.to(device)
print(f"[preflight] align batch x{tuple(xa.shape)} img{tuple(ia.shape)} "
      f"labels_kept={(ya != -100).sum().item()}", flush=True)
for i in range(2):
    loss = criterion(model(xa, ia).view(-1, 50257), ya.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"[preflight] align step {i+1} align_ce {loss.item():.4f}", flush=True)

# ---- the per-epoch eval path ----------------------------------------------
model.eval()
ce, tok = eval_stream(model, dl_wiki_test, device, with_img=False, limit=2048)
print(f"[preflight] eval wiki: ce {ce:.4f} ppl {math.exp(ce):.2f} ({tok:,} tok)", flush=True)
ce, tok = eval_stream(model, dl_align_test, device, with_img=True, limit=2048)
print(f"[preflight] eval align: ce {ce:.4f} ({tok:,} tok)", flush=True)

print("[preflight] ALL GREEN - nothing written to check_point/", flush=True)
