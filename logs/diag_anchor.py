"""Is the contrastive term being fed a constant? Diagnose the retrieval anchors.

Run from anywhere (paths are resolved from this file's location):

    python script/diag_anchor.py check_point/gpt2_model_best.pth 2048
    # argv[1] = checkpoint (raw state_dict), optional
    # argv[2] = how many align-test samples to probe (default 2048)

No training, no new data: one forward over a fixed slice of the align TEST split,
then the same 2048-pool retrieval scored under several pooling / normalisation
variants. It separates three failures that all look like "R@1 is at the noise
floor":

  (1) the information is not in the representation at all
      -> every variant stays at 1/N
  (2) the information is there but mean(49) averages it away
      -> mean is at 1/N, mean-minus-DC / last-token are far above it
  (3) the image slot is a leaky copy of the text side
      -> the blind text anchor (img=None forward) collapses while the seeing one
         stays high

Also prints cos(pooled image slots, text anchor hidden). Mean-pooling 49 highly
correlated tokens keeps the shared direction at full amplitude while the
image-specific residual decays by sqrt(49), so that cosine climbing towards 1 is
the signature of a pooled feature that no longer says which image it came from.
"""
import os
import sys

import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from model.gpt2 import GPT2                      # noqa: E402
import prompt.prompt_vision_align as pv          # noqa: E402

SLOTS = 49
TEST_JSON = os.path.join(REPO, "data/pre-training/vision_alignment/test.jsonl")


def pairwise_cos(x):
    """Mean / std of the off-diagonal cosine: how much two different images differ."""
    x = nn.functional.normalize(x, dim=-1)
    s = x @ x.t()
    eye = torch.eye(s.size(0), dtype=torch.bool, device=s.device)
    off = s[~eye]
    return float(off.mean()), float(off.std())


def spectrum(x):
    """How much of the between-image variance sits in ONE direction (the DC)."""
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    v = s ** 2
    frac1 = float(v[0] / v.sum())
    part = float(v.sum() ** 2 / (v ** 2).sum())      # participation ratio, 1 = rank 1
    return frac1, part


def rank_stats(sim):
    """sim: (N, N), the diagonal being the true pair."""
    r = (sim > sim.diagonal()[:, None]).sum(1)
    return dict(r1=float((r == 0).float().mean()),
                r5=float((r < 5).float().mean()),
                mean_rank=float(r.float().mean()),
                n=r.numel())


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2().to(dev)
    if ckpt:
        sd = torch.load(ckpt, map_location=dev)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[ckpt] {ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            print(f"[ckpt] missing: {sorted(missing)[:6]}{' ...' if len(missing) > 6 else ''}")
    else:
        print("[ckpt] none given -> freshly initialised model (baseline is 1/N)")
    model.eval()
    scale = float(model.logit_scale.detach().exp())
    print(f"[ckpt] exp(logit_scale) = {scale:.2f}   (1/tau; clamp upper bound is 100.0)")

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("json", data_files=TEST_JSON, split=f"train[:{n_samples}]")
    x, y, anchor, paths = pv._encode_split(ds, enc)
    dl = DataLoader(pv.VisionAlignDataset(x, y, anchor, paths, pv._image_transform()),
                    batch_size=128, shuffle=False)
    print(f"[data] {len(paths)} test samples, {len(dl)} batches")

    P = {k: [] for k in ("mean", "max", "last", "mean_dc", "mean_z")}
    T_see, T_blind, cos_pass = [], [], []
    with torch.no_grad():
        for xb, yb, img, anc in dl:
            xb, img, anc = xb.to(dev), img.to(dev), anc.to(dev)
            _, h = model(xb, img, True)              # text side CAN see the image
            _, hb = model(xb, None, True)            # text side blind
            slots = h[:, :SLOTS]
            P["mean"].append(slots.mean(1))
            P["max"].append(slots.max(1).values)
            P["last"].append(slots[:, -1])
            ar = torch.arange(xb.size(0), device=dev)
            T_see.append(h[ar, anc])
            T_blind.append(hb[ar, anc])
            cos_pass.append(nn.functional.cosine_similarity(slots.mean(1), h[ar, anc], dim=-1))

    # DC removal uses the statistics of this fixed probe pool, so it is a property
    # of the measurement, not of the training loop.
    pm = torch.cat(P["mean"])
    mu = pm.mean(0, keepdim=True)
    sd_ = pm.std(0, keepdim=True) + 1e-6
    for k in ("mean", "max", "last"):
        P[k] = [torch.cat(P[k])]
    P["mean_dc"] = [pm - mu]
    P["mean_z"] = [(pm - mu) / sd_]
    T_see, T_blind, cos_pass = torch.cat(T_see), torch.cat(T_blind), torch.cat(cos_pass)

    print(f"\n[leak check] cos(pooled image slots, text anchor hidden): "
          f"mean {float(cos_pass.mean()):.3f}  median {float(cos_pass.median()):.3f}")
    print("             (>0.9 -> the trunk is copying the image into the text positions)")

    def score(img_feat, txt_feat, tag, txt_tag):
        i = nn.functional.normalize(model.con_img(img_feat), dim=-1)
        t = nn.functional.normalize(model.con_txt(txt_feat), dim=-1)
        sim = i @ t.t()
        st = rank_stats(sim)
        print(f"  {tag:<8s} x {txt_tag:<5s} R@1 {st['r1']:.4f}  R@5 {st['r5']:.4f}  "
              f"mean_rank {st['mean_rank']:7.1f} (random {st['n']/2:.0f})  "
              f"sim range [{float(sim.min()):.3f}, {float(sim.max()):.3f}]")

    print("\n[variants] image-side pooling x text-side anchor")
    for k in ("mean", "max", "last", "mean_dc", "mean_z"):
        m_, s_ = pairwise_cos(P[k][0])
        f1, pr = spectrum(P[k][0])
        print(f"\n[{k}] pairwise cos {m_:.3f} +/- {s_:.3f} | PC1 explains {f1:.1%} of "
              f"between-image variance | participation ratio {pr:.1f}")
        score(P[k][0], T_see, k, "see")
        score(P[k][0], T_blind, k, "blind")

    print("\n[PREDICTIONS if 'mean-pool feeds a constant' is right]")
    print("  mean       pairwise cos 0.9-0.99, PC1 high, R@1 at the noise floor")
    print("  mean_z     pairwise cos ~0, R@1 clearly higher than mean -> info was there")
    print("  text=blind R@1 collapses to 1/N -> no leak; stays high -> leak exists")
    print("  all variants at 1/N -> the representation has no discriminative content;")
    print("  the fix is training-side (anchor placement / lambda / augmentation),")
    print("  not geometric.")


if __name__ == "__main__":
    main()
