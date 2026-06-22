# GPT-2 from Scratch

<<<<<<< HEAD
An 85M-parameter decoder-only transformer implemented from scratch in PyTorch, trained on WikiText-2.
=======
An ~49M-parameter decoder-only transformer implemented from scratch in PyTorch, trained on WikiText-2.
>>>>>>> 2dd0673 (Initial commit: GPT-2 from scratch (~49.4M params, wikitext-2))

## Architecture

| Parameter | Value |
|-----------|-------|
| d_model (embedding dim) | 384 |
| n_heads | 6 |
| n_layers | 6 |
| block_size (context) | 256 |
<<<<<<< HEAD
| Total params | ~85M |
=======
| Total params | ~49.4M |
>>>>>>> 2dd0673 (Initial commit: GPT-2 from scratch (~49.4M params, wikitext-2))

## Features

- **100% from scratch** — no HuggingFace `GPT2LMHeadModel` or `Trainer`, pure `nn.Module` + custom training loop
- **tiktoken** GPT-2 tokenizer
- **WikiText-2** dataset via `datasets` library
- **Auto device detection** — CUDA → MPS → CPU fallback (no hardcoded `mps`)
<<<<<<< HEAD
=======
- **Untied LM head** — separate Linear output projection (not weight-tied with token embedding), adding ~19M capacity for next-token prediction
>>>>>>> 2dd0673 (Initial commit: GPT-2 from scratch (~49.4M params, wikitext-2))
- **Multi-GPU ready** — designed for distributed training on GPU clusters

## Files

- `gpt2.py` — full implementation: model, dataset, training loop
- `requirements.txt` — dependencies

## Quick Start

```bash
pip install -r requirements.txt
python gpt2.py
```

The script auto-detects your device and starts training on WikiText-2.
