# GPT-2 + ViT: a small multimodal model written from scratch

An 86M-parameter GPT-2 decoder and a 4-layer vision transformer, both written by
hand in PyTorch, trained as one model: 49 visual tokens are injected into the
first 49 positions of the token sequence, so a single causal decoder serves both
plain text and image-conditioned generation.

No `transformers`, no `trl`, no pre-trained weights. The decoder, the ViT, the
attention, the interleaved training loop and the evaluation harness are all in
this repository. That is the point: when the model produces a caption that is
plausible but wrong, or a retrieval score that looks too good, there is no
third-party layer to blame and no layer hiding the cause.

```
text      <|endoftext|> computer science is the study of ... <|endoftext|>
image     [v1] [v2] ... [v49] <|endoftext|> What is this? # a red apple <|endoftext|>
          \_________________/
           49 slots, overwritten with projected ViT output before the decoder runs
```

## Architecture

| component | details |
|---|---|
| decoder | 6 layers, d_model 512, 8 heads, `block_size` 320, vocab 50,257 (tiktoken `gpt2`), GELU FFN 4x, pre-norm, learned positional embeddings, causal mask |
| vision encoder | ViT: 224x224 input, 32x32 patches -> 7x7 = **49 tokens**, dim 512, 8 heads, 4 blocks, MLP 4x, learned positional embeddings |
| bridge | `projector`: Linear(512,512) -> GELU -> Linear(512,512) -> LayerNorm |
| blind path | `no_image_emb`, a learned (1,49,512) tensor used when `img=None` |
| contrastive head | `con_img` / `con_txt`: Linear(512,512) -> GELU -> Linear(512,256), plus a learnable logit scale clamped to tau ∈ [0.05, 0.10] |
| total | 86.1M parameters |

The image path is a token-slot substitution, not a cross-attention branch:

1. `token_emb(x)` is computed as usual.
2. `token_emb[:, :49, :]` is overwritten with `projector(vit(image))`, or with
   `no_image_emb` when there is no image.
3. The result goes through the standard causal transformer.

Because attention is causal, positions 0..48 can only see the image slots and
each other, and text positions can see everything before them. The image slots
therefore cannot be polluted by the caption — a property the contrastive
objective depends on (see "What the numbers taught" below).

## Training pipeline

| stage | data | objective | status |
|---|---|---|---|
| 1. text pretraining | Simple English Wikipedia, 338,479 articles, ~39M supervised tokens | next-token CE, 49 slots masked out | done |
| 2. image-text alignment | LLaVA-Pretrain (`blip_laion_cc_sbu_558k`): 502,315 train / 55,813 test image-caption pairs | caption CE + 0.5 x symmetric in-batch InfoNCE | done, 5 epochs |
| 3. VQA fine-tuning | VQAv2 (planned): short closed-set answers, 3,129-answer label space | answer CE, evaluated as accuracy | in progress |

**Interleaving.** Stages 1 and 2 run together: one alignment batch every `K` text
batches. `K` is not a learning-rate knob — it sets what fraction of the updates
go to alignment (`K=1` -> 50%, `K=2` -> 33%, `K=4` -> 20%) and therefore how many
passes over the 502k image-caption pairs the run makes. The reported run uses
`K=2`, which is 4.5 passes in five epochs. Overfitting in this stream starts
around 2-3 passes, not at a fixed epoch count.

**Why a contrastive term at all.** Caption CE saturates: with ~13 supervised
tokens per pair, the model can take most of the loss by learning the LAION
caption style ("a photo of ..."), and the image only explains the remainder. The
InfoNCE term removes everything shared by every caption in the batch (it
contributes equally to all negatives and cancels in the softmax), leaving only
what distinguishes *this* image's caption from the others.

Hardware: 4x RTX A6000, shared, no scheduler — a single GPU is selected per run.

## Results

Text stream, five epochs at `K=2`:

| epoch | wiki ppl | align CE (test) |
|---|---|---|
| 3 | 37.71 | 3.881 |
| 4 | 35.87 | 3.838 |
| 5 | **34.73** | **3.831** |

Retrieval over the full 55,813-pair test pool (blind text anchor, see below):

| epoch | R@1 | R@5 | mean rank | mean rank as % of pool |
|---|---|---|---|---|
| 3 | 0.62% | 3.6% | 2,901 | 5.2% |
| 4 | 0.77% | 4.4% | 2,626 | 4.7% |
| 5 | **0.93%** | **4.9%** | **2,521** | **4.5%** |

Three-way probe on 4,096 held-out pairs, same checkpoint, same token layout —
only the image changes:

| condition | caption CE | perplexity |
|---|---|---|
| matched image | **3.846** | 46.8 |
| mismatched image (derangement) | 4.499 | 89.9 |
| no image at all (`img=None`) | 4.423 | 83.3 |

The image is worth 0.58 nats over the language prior, and a *wrong* image is
worse than no image — the model reads the pixels and is misled by them, which is
the behaviour you want to see.

## What the numbers taught

**1. The contrastive objective had a trivial solution.** Taking the text anchor
from a forward pass that could see the image let the two projection heads agree
by routing image information into a caption position, without the vision encoder
learning anything: R@1 = 0.99 with mean rank 0. Re-anchoring the text side on an
`img=None` forward dropped the same checkpoint to R@1 = 0.0015. Every retrieval
number above is measured with the blind anchor.

**2. Gap-shaped metrics inflate with model confidence.** "CE with a broken image
minus CE with the right image" looks like a direct measure of how much the vision
path contributes. Between epoch 4 and 5 that gap grew from 0.59 to 0.65 nats,
while the matched-image CE did not move (3.8500 -> 3.8464, inside the sampling
noise of 4,096 samples). The gap grew because the model got more confident, not
because it saw better. Gaps are only comparable across checkpoints if the
"good" side of the gap is itself improving.

**3. Passes over the data, not epochs, decide when overfitting starts.** With
`K=8` the alignment stream saw 1.69 passes in five epochs and never overfit; with
`K=1` it saw 2.70 passes in the first epoch and started memorising. Train/test
alignment CE diverged 0.39 -> 0.50 nats between epochs 4 and 5 (train improved 18x
faster than test), which is where the run was stopped.

## Honest limitations

- **Caption generation is attribute-level.** Given an apple, the model produces
  `red and white circle round mug`, then `a red plastic water hose`, then
  `cherry zip pouch`. Colour and shape survive across checkpoints; the object
  class is wrong every time. ~13 supervised tokens per pair is the ceiling here.
- **The text side has form without facts.** At perplexity 34.7 it writes
  Wikipedia-shaped prose with invented content. Fine for short closed-set VQA
  answers, not for open generation.
- **Retrieval is still weak in absolute terms.** R@1 of 0.93% is not useful on
  its own; the honest summary is "the correct caption lands in the top 4.5% of a
  55,813-item pool".
- **The ViT attention uses dropout 0.1**, so a train-mode forward pass is not a
  pure function of its input. Every evaluation path runs in `eval()` mode for
  exactly this reason; a self-check that compares two train-mode forwards will
  see a difference of ~0.27 on this model.
- **The alignment dataloader is not portable.** `prompt/prompt_vision_align.py`
  locates `images.zip` by globbing a HuggingFace cache path; it assumes the
  dataset was downloaded through `datasets`.
- **Stage 3 is not finished.** `SFT_train.py` is a scaffold: it still targets
  `trl`/`transformers` APIs that do not apply to a hand-written model.

## Repository layout

```
Main.py                        entry point: reads a YAML config, trains, then runs two generations
Train_and_Eval.py              interleaved text/align training, evaluation, checkpointing
SFT_train.py                   stage-3 scaffold (not wired up yet)
config/pre-training/           YAML: epochs, learning rate, K
model/gpt2.py                  decoder, projector, no_image_emb, InfoNCE head
model/vision_encoder.py        ViT
model/generator.py             sampling for both text-only and image-conditioned generation
prompt/prompt_wiki.py          text stream
prompt/prompt_vision_align.py  alignment stream (images decoded lazily from images.zip)
prompt/prompt_vision.py        VQA/SFT stream
script/                        dataset preprocessing, anchor diagnostic
logs/                          probe scripts that produced the numbers above
docs/PROJECT_LOG.zh.md         working log (Chinese): every wrong turn, in order
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python Main.py            # then type the config path, e.g. config/pre-training/contrast.yaml
```

Expected data layout (paths are relative to the repository root):

```
data/pre-training/wiki/{train,test}.json
data/pre-training/vision_alignment/{train,test}.jsonl
data/pre-training/vision_alignment/  -> images.zip via the HuggingFace cache
data/post-training/pet/cat_dog.json
check_point/gpt2_model_best.pth      raw state_dict, loaded directly by model/generator.py
```

A YAML config carries three keys — `train.epoch`, `train.learning_rate`,
`train.K`. Batch sizes and stream locations are hard-coded in the `prompt/`
modules on purpose: a config key that nothing reads is worse than a missing key.

## Roadmap

1. VQAv2 fine-tuning with a paired evaluation: the same questions answered with
   the matched image, a mismatched image, and no image. The deliverable is the
   accuracy gap over the blind baseline, not an absolute number — VQAv2 gives a
   text-only model ~40% for free.
2. If that gap is small: 16x16 patches plus a resampler (the 49-slot interface
   stays, so existing text weights keep loading), image augmentation, and ViT
   dropout tuning — not more alignment epochs, which the epoch-5 run showed no
   longer buys anything measurable.
3. Only then: instruction-style SFT and a demo UI.
