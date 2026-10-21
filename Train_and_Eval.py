import torch
import torch.nn as nn
from model.gpt2 import GPT2
import prompt.prompt_wiki
import prompt.prompt_vision
import prompt.prompt_vision_align
import json
import os
import math
from tqdm import tqdm
try:
    import wandb
except ImportError:
    # local debugging on a machine without wandb installed: the run degrades to
    # the plain stdout logging instead of dying at import time
    wandb = None

# ---- checkpoint layout -------------------------------------------------------
# gpt2_model_best.pth is a RAW state_dict on purpose: model/generator.py loads it
# straight with model.load_state_dict(torch.load(...)). Everything needed to
# continue a run lives in check_point_current.pth instead, so inference and
# resume do not have to agree on one file format.
CKPT_DIR = "check_point"
BEST_WEIGHTS = os.path.join(CKPT_DIR, "gpt2_model_best.pth")        # raw state_dict
CURRENT_WEIGHTS = os.path.join(CKPT_DIR, "check_point_current.pth")  # full training state
BEST_JSON = os.path.join(CKPT_DIR, "check_point_best.json")
CURRENT_JSON = os.path.join(CKPT_DIR, "check_point_current.json")

SAVE_EVERY = 500              # steps between mid-epoch checkpoints
LOG_EVERY = 200               # steps between progress lines
BEST_METRIC_NAME = "wiki_ppl"  # the model-selection criterion
# Picking "the best R@1" is a max over a noisy number. The margin has to be sized
# to the evaluation pool: on the full 55,813-sample test set the binomial noise at
# R@1 = 0.2% is sqrt(p(1-p)/N) ~ 0.0002, so 0.0005 is roughly 2 sigma -- big
# enough to reject a lucky epoch, small enough that a real move still gets in.
R1_MARGIN = 0.0005

# ---- contrastive term on the align stream ------------------------------------
# The align objective is  text_ce + LAMBDA_CON * infonce.
LAMBDA_CON = 0.5

WANDB_PROJECT = "gpt2-vit"     # dashboard project name
USE_WANDB = wandb is not None  # False -> nothing is sent, stdout logging still runs
# sidecar file holding the wandb run id, so a restarted job rejoins its own run
WANDB_ID_FILE = os.path.join(CKPT_DIR, "wandb_run_id.txt")


def atomic_save(obj, path):
    """Save through a temp file, then rename.

    A job killed in the middle of torch.save leaves a truncated file that can no
    longer be loaded - which is how a run that has been going for days loses its
    checkpoint on a shared machine. os.replace() is atomic on the same filesystem.

    The tmp name carries the pid. With a fixed name, two Main.py processes on the
    same checkout share one tmp path: A writes it, B's os.replace() renames it
    away, and A's os.replace() then dies with
    "No such file or directory: ...pth.tmp -> ...pth" - which looks like a disk
    or corruption problem but is really just the two runs stealing from each other.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_arch(model):
    """Architecture fingerprint, written into the checkpoint.

    Two of these numbers are load-bearing for a checkpoint: vocab_size decides the
    shape of token_emb / head, block_size decides pos_emb rows + causal_mask.
    Storing them means a mismatch reports what changed instead of a bare
    "size mismatch for pos_emb.weight".
    """
    return {
        "vocab_size": model.token_emb.num_embeddings,
        "d_model": model.token_emb.embedding_dim,
        "n_heads": model.layers[0].n_heads,
        "n_layers": len(model.layers),
        "block_size": model.block_size,
    }


def eval_stream(model, dataloader, device, with_img, limit=None):
    """Token-weighted cross-entropy over one dataloader.

    Weighted by real tokens, not by batch: every sample has a different number of
    unmasked positions (the padding is -100), so averaging the per-batch means
    would be biased towards the short samples.
    """
    total_nll = 0.0
    total_tok = 0
    all_img_feat =[]
    all_text_feat= []
    for i, item in enumerate(dataloader):
        if limit is not None and i * dataloader.batch_size >= limit:
            break
        if with_img:
            x, y, img, anchor_test = item
            with torch.no_grad():
                logits, hidden = model(x.to(device), img.to(device), True)
                # The text anchor must come from a forward that cannot see the image.
                # Taking it from the image-visible forward makes the retrieval score
                # rank 0 for every sample without the vision encoder having learned
                # anything -- measured: R@1 0.9912 visible vs 0.0015 blind, on the
                # same checkpoint (script/diag_anchor.py).
                _, hidden_blind = model(x.to(device), None, True)
                anchor_test = anchor_test.to(device)
            img_feat, _ = model.align_features(hidden, anchor_test)
            _, text_feat = model.align_features(hidden_blind, anchor_test)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

            all_img_feat.append(img_feat)
            all_text_feat.append(text_feat)

        else:
            x, y = item
            logits = model(x.to(device))
        y = y.to(device)
        total_nll += nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            ignore_index=-100, reduction="sum").item()
        total_tok += (y != -100).sum().item()

    if with_img:
        all_img_feats = torch.cat(all_img_feat, dim=0)    # [N, 256], L2 normalised
        all_txt_feats = torch.cat(all_text_feat, dim=0)   # [N, 256], L2 normalised
        N = all_img_feats.shape[0]
        # Rank of the true pair, computed in chunks: the full N x N similarity
        # matrix is 12.5 GB at N = 55,813 and only the row-wise rank is needed.
        # Ties count as hits ("strictly more similar" is what moves you down).
        true_sim = (all_img_feats * all_txt_feats).sum(1)
        rank_parts = []
        for s in range(0, N, 2048):
            sim = all_img_feats[s:s + 2048] @ all_txt_feats.t()
            rank_parts.append((sim > true_sim[s:s + 2048, None]).sum(1))
        ranks = torch.cat(rank_parts)
        r1 = (ranks == 0).float().mean().item()
        r5 = (ranks < 5).float().mean().item()
        # mean rank is the one with usable resolution: on a 55,813 pool its standard
        # error is ~2.5 ranks, versus R@1's ~1 hit, so a real improvement shows up.
        mean_rank = ranks.float().mean().item()
        return (total_nll / max(total_tok, 1), total_tok, r1, r5, mean_rank)
    return total_nll / max(total_tok, 1), total_tok


def train_and_eval(epochs_num, learning_rate,k):
    #create the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(CKPT_DIR, exist_ok=True)

    model = GPT2().to(device)
    # the optimizer must exist before its state_dict can be loaded back into it
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 3. define the loss function
    criterion = nn.CrossEntropyLoss()

    # ---- resume from the newest checkpoint ----------------------------------
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    best_r1 = -1
    if os.path.exists(CURRENT_WEIGHTS):
        ckpt = torch.load(CURRENT_WEIGHTS, map_location=device)
        if ckpt["arch"] != make_arch(model):
            raise RuntimeError(
                "checkpoint was written by a different architecture:\n"
                f"  in checkpoint {ckpt['arch']}\n"
                f"  current model {make_arch(model)}\n"
                f"Move {CKPT_DIR}/ aside to start from scratch.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_metric = ckpt["best_metric"]
        # .get: checkpoints written before best_r1 existed do not carry the key,
        # and a run that resumes must not silently restart the R@1 comparison from 0
        best_r1 = ckpt.get("best_r1_align", 0.0)
        print(f"[resume] {CURRENT_WEIGHTS}: finished epoch {ckpt['epoch']+1}, "
              f"global_step {global_step}, best_{BEST_METRIC_NAME} "
              f"{best_metric:.2f} -> training epoch {start_epoch+1}..{epochs_num}")
    else:
        print(f"[train] no checkpoint at {CURRENT_WEIGHTS}, starting from scratch")

    if start_epoch >= epochs_num:
        print(f"[done] {epochs_num} epochs already finished, nothing to do")
        return

    # maybe need a signal to distinguish pre-training and post-training, but for now, i am writing for pretraining wiki+align
    dataloader_train_wiki, dataloader_test_wiki = prompt.prompt_wiki.build_wiki_dataloader()

    dataloader_train_align, dataloader_test_align = prompt.prompt_vision_align.get_dataloader_vision_align()
    # one align batch every K text batches
    if k=="None":
        K = max(1, round(epochs_num * len(dataloader_train_wiki) / len(dataloader_train_align)))
    else:
        K= int(k)
    # created OUTSIDE the epoch loop: the align stream keeps walking through the
    # 502k images from epoch to epoch, so K is what decides how many passes over
    # the alignment data the whole run sees.
    it_align_train = iter(dataloader_train_align)

    # ---- wandb: open the run -------------------------------------------------
    # Placed AFTER the "nothing to do" return above, so a job that is re-launched
    # with the training already finished does not open an empty run on the board.
    # The id comes from a sidecar file next to the checkpoints: a preempted job
    # that is restarted rejoins its OWN curve. Deleting check_point/wandb_run_id.txt
    # (or the whole check_point/) makes the next start a brand-new run.
    if USE_WANDB:
        run_id = None
        if os.path.exists(WANDB_ID_FILE):
            with open(WANDB_ID_FILE, "r", encoding="utf-8") as f:
                run_id = f.read().strip() or None
        wandb.init(project=WANDB_PROJECT,
                   name=f"wiki+align-lr{learning_rate}-ep{epochs_num}",
                   id=run_id, resume="allow",
                   config={"epochs": epochs_num,
                           "learning_rate": learning_rate,
                           "batch_size_wiki": dataloader_train_wiki.batch_size,
                           "steps_per_epoch": len(dataloader_train_wiki),
                           "align_every_K": K,
                           "save_every": SAVE_EVERY,
                           "log_every": LOG_EVERY,
                           "resumed_from_epoch": start_epoch + 1,
                           **make_arch(model)})
        if run_id is None:
            with open(WANDB_ID_FILE, "w", encoding="utf-8") as f:
                f.write(wandb.run.id)
        print(f"[wandb] {wandb.run.url} mode={wandb.run.settings.mode}", flush=True)

    # 4. training loop
    epochs = epochs_num
    for epoch in range(start_epoch, epochs):
        model.train()
        text_sum = text_n = align_sum = align_n = 0
        align_loss_step = None

        # disable=None -> bar only when stderr is a terminal, so "nohup python
        # Main.py > train.log" does not fill the log file with carriage-return frames
        pbar = tqdm(dataloader_train_wiki, total=len(dataloader_train_wiki),
                    desc=f"epoch {epoch+1}/{epochs}", unit="step", disable=None)
        for step, (x, y) in enumerate(dataloader_train_wiki):
            x, y = x.to(device), y.to(device)
            text_loss = criterion(model(x).view(-1, 50257), y.view(-1))
            optimizer.zero_grad()
            text_loss.backward()
            optimizer.step()
            # loss.item() is a mean over the unmasked tokens, so scale it back by
            # the token count to accumulate a token-weighted epoch average
            text_sum += text_loss.item() * (y != -100).sum().item()
            text_n   += (y != -100).sum().item()

            if step % K == 0:
                try:
                    xa, ya, ia, anchor = next(it_align_train)
                except StopIteration:                  # if shuffle done, then restart
                    it_align_train = iter(dataloader_train_align)
                    xa, ya, ia, anchor = next(it_align_train)
                xa, ya, ia = xa.to(device), ya.to(device), ia.to(device)
                anchor = anchor.to(device)
                # one forward gives both objectives: the logits for the caption CE
                # and the pre-head hidden for the contrastive term.
                logits, hidden = model(xa, ia, return_hidden=True)
                align_ce = criterion(logits.view(-1, 50257), ya.view(-1))
                # The text anchor comes from a forward that CANNOT see the image.
                # With the image-visible forward the contrastive term has a trivial
                # solution -- route the image into a caption position, then let the
                # two projection heads map both sides onto the same point -- and it
                # reaches R@1 0.99 / mean rank 0 while the vision encoder learns
                # nothing. script/diag_anchor.py measures that gap on a checkpoint:
                # R@1 0.9912 visible vs 0.0015 blind, same weights.
                _, hidden_blind = model(xa, None, return_hidden=True)
                img_feat, _ = model.align_features(hidden, anchor)
                _, txt_feat = model.align_features(hidden_blind, anchor)
                con_loss = model.infonce(img_feat, txt_feat)
                align_loss = align_ce + LAMBDA_CON * con_loss
                optimizer.zero_grad()
                align_loss.backward()
                optimizer.step()
                # accumulated/logged as the plain CE, so the align_ce numbers stay
                # comparable with the runs that predate the contrastive term
                align_sum += align_ce.item() * (ya != -100).sum().item()
                align_n   += (ya != -100).sum().item()
                align_loss_step = align_ce.item()

            global_step += 1

            if global_step % LOG_EVERY == 0:
                line = (f"  epoch {epoch+1} step {step}/{len(dataloader_train_wiki)} "
                        f"text_ce {text_loss.item():.4f}")
                if align_loss_step is not None:
                    line += f" align_ce {align_loss_step:.4f}"
                print(line, flush=True)

                pbar.set_postfix(text_ce=f"{text_loss.item():.4f}")
                if USE_WANDB:
                    # x-axis is global_step, not the per-epoch step index, so the
                    # curve stays continuous across epochs and restarts
                    payload = {"train/text_ce": text_loss.item()}
                    if align_loss_step is not None:
                        payload["train/align_ce"] = align_loss_step
                        align_loss_step = None          # consumed
                    wandb.log(payload, step=global_step)

            if global_step % SAVE_EVERY == 0:
                # mid-epoch checkpoint: keeps epoch=<current>, so after a crash the
                # interrupted epoch is replayed from its start (the step index is
                # not persisted). Costs at most one duplicated epoch, keeps the
                # loop simple.
                atomic_save({"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "epoch": epoch,
                             "global_step": global_step,
                             "best_metric": best_metric,
                             "best_r1_align": best_r1,
                             "arch": make_arch(model)}, CURRENT_WEIGHTS)
        pbar.close() 

        # evaluate seperately: never average the two streams into one number.
        # wiki_ppl is the criterion (that is the ability worth protecting);
        # align_ce is a monitor - if it drops while wiki_ppl rises, K is too small.
        # add R@1
        model.eval()

        wiki_ce, wiki_tok = eval_stream(model, dataloader_test_wiki, device, with_img=False)
        # full test pool (no limit): mean rank is the trend number and it needs the
        # resolution -- 55,813 samples put its standard error at ~2.5 ranks
        align_ce, align_tok, r1_align, r5_align, mean_rank_align = eval_stream(
            model, dataloader_test_align, device, with_img=True)
        wiki_ppl = math.exp(wiki_ce)
        train_text_ce = text_sum / max(text_n, 1)
        train_align_ce = align_sum / max(align_n, 1)
        print(
            f"[epoch {epoch+1}/{epochs}] train text_ce {train_text_ce:.4f} "
            f"align_ce {train_align_ce:.4f} | "
            f"test wiki_ppl {wiki_ppl:.2f} ({wiki_tok:,} tok) "
            f"align_ce {align_ce:.4f} ({align_tok:,} tok) "
            f"r1 {r1_align:.4f} r5 {r5_align:.4f} mean_rank {mean_rank_align:.1f} "
            f"(best r1 {best_r1:.4f})", flush=True)

        # ---- checkpoints -----------------------------------------------------
        # wiki_ppl is the gate: only an epoch that really improved the text side is
        # allowed to become the best model. Among those, the highest R@1 wins. So a
        # worse-aligned epoch never overwrites the file, and an abnormal (NaN/inf)
        # ppl epoch saves nothing -- which is what the ppl-only rule did anyway, so
        # that path keeps its old behaviour.
        ppl_ok = math.isfinite(wiki_ppl)
        if not ppl_ok:
            print(f"  [warn] wiki_ppl is not finite ({wiki_ppl}); "
                  f"{BEST_WEIGHTS} left untouched", flush=True)
        is_best = ppl_ok and wiki_ppl < best_metric
        if is_best:
            best_metric = wiki_ppl
            if r1_align > best_r1 + R1_MARGIN:
                best_r1 = r1_align
                atomic_save(model.state_dict(), BEST_WEIGHTS)
                with open(BEST_JSON, "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch,
                               "global_step": global_step,
                               "r1_align": r1_align,
                               "best_r1_align": best_r1,
                               "wiki_ppl": wiki_ppl,
                               "arch": make_arch(model)}, f, indent=2)
                print(f"  -> new best: {BEST_METRIC_NAME} {wiki_ppl:.2f}, "
                      f"r1 {r1_align:.4f}, saved {BEST_WEIGHTS}", flush=True)

        if USE_WANDB:
            wandb.log({"train/text_ce_epoch": train_text_ce,
                       "train/align_ce_epoch": train_align_ce,
                       "test/wiki_ce": wiki_ce,
                       "test/wiki_ppl": wiki_ppl,
                       "test/align_ce": align_ce,
                       "test/r1_align": r1_align,
                       "test/r5_align": r5_align,
                       "test/mean_rank_align": mean_rank_align,
                       "best/wiki_ppl": best_metric,
                       "epoch": epoch + 1}, step=global_step)

        atomic_save({"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "epoch": epoch,
                     "global_step": global_step,
                     "best_r1_align": best_r1,
                     "best_metric": best_metric,
                     "arch": make_arch(model)}, CURRENT_WEIGHTS)

        with open(CURRENT_JSON, "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch,
                       "global_step": global_step,
                       "r1_align": r1_align,
                       "r5_align": r5_align,
                       "mean_rank_align": mean_rank_align,
                       "wiki_ppl": wiki_ppl,
                       "align_ce": align_ce,
                       "train_text_ce": train_text_ce,
                       "train_align_ce": train_align_ce,
                       "best_r1_align": best_r1,
                       "best_metric": best_metric,
                       "arch": make_arch(model)}, f, indent=2)
        
    # wandb also closes on process exit; doing it here guarantees the last points
    # are uploaded before the script moves on to generator.generate() in Main.py
    if USE_WANDB:
        url = wandb.run.url
        wandb.finish()
        print(f"[wandb] run closed: {url}", flush=True)
