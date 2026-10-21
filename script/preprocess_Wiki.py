"""
Build the wiki knowledge corpus for prompt_wiki.py.

Source : wikimedia/wikipedia -- config "20231101.simple" (Simple English Wikipedia)
Output : data/wiki/train.json, data/wiki/test.json   (JSON array of {"text": ...})

Why this replaces the previous attempt
--------------------------------------
The old script split every article on "\n" and kept every line with >= 20 chars.
That pulled in section headings, the category-link dump at the bottom of each
article, track listings and flattened table rows. Measured on the old output:
54% of the 2.0M samples were under 10 words, 51% had no sentence punctuation,
26% were duplicates ("American movie actors" alone appeared 4321 times).
The source was fine -- the line-level split was the bug.

What this script does
---------------------
1. Stop reading at the first trailing link section (Other websites / References /
   Related pages / ...): everything after it is categories and link lists.
2. Keep only real prose. A line survives if it has sentence-ending punctuation,
   is not a list item / table row / track listing, and is not a stub.
   Section headings are killed for free because they carry no final punctuation.
3. Split articles 98/2 FIRST (fixed seed) so no article appears in both splits.
4. Re-pack the surviving sentences of each article into passages of at most
   MAX_TEXT_TOKEN gpt2 tokens, cut on sentence boundaries only. The article
   title is prefixed to the first passage so a passage is self-contained
   (the old data had passages starting with "He became King of ...").
5. Global de-duplication.

Run from anywhere:
    python script/preprocess_Wiki.py
"""

import json
import os
import random
import re

import tiktoken
from datasets import load_dataset

# --------------------------------------------------------------------- paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WIKI_DIR = os.path.join(PROJECT_ROOT, "data", "pre-training", "wiki")
TRAIN_OUT = os.path.join(WIKI_DIR, "train.json")
TEST_OUT = os.path.join(WIKI_DIR, "test.json")

# --------------------------------------------------------------------- knobs
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.simple"
MAX_TEXT_TOKEN = 200      # one passage must fit the text budget of the frame
MIN_WORDS = 5             # below this it is a stub, not a sentence
TEST_RATIO = 0.02
SEED = 42

# A section whose body is a list of links / categories, never knowledge.
DROP_SECTION = re.compile(
    r"^(other websites|external links|references|related pages|see also|"
    r"notes|further reading|sources|footnotes|gallery)$",
    re.IGNORECASE,
)
# Lines that start with a list marker ( "* item", "1. item", "A. item", "- item" ).
LIST_MARKER = re.compile(r"^([*#]|[--\u2013\u2014]|[A-Za-z0-9]{1,3}\.)\s")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
LINE_END = re.compile(r"[.!?]$")          # a whole line must end as a sentence
CITE_TAIL = re.compile(r"(\[\d+\])+\s*$")  # trailing citation markers "[1]"
TRACK_LISTING = re.compile(r"\d+:\d{2}")          # "4:16", "12:03" -> discography


def prose_sentences(text):
    """Split an article body into sentences, dropping headings/lists/tables."""
    out = []
    for raw in text.split("\n"):
        line = CITE_TAIL.sub("", raw.strip()).strip()
        if not line:
            continue
        if DROP_SECTION.match(line):
            break                                  # link sections sit at the end
        if "|" in line:                            # table row
            continue
        if LIST_MARKER.match(line):                # bullet / numbered / lettered item
            continue
        if TRACK_LISTING.search(line):             # track listing
            continue
        if len(line.split()) < MIN_WORDS:          # stub or heading
            continue
        if not LINE_END.search(line):              # heading, or a bibliography row:
            continue                               # neither ends as a sentence
        for sent in SENT_SPLIT.split(line):
            sent = sent.strip()
            if sent and sent[-1] in ".!?":         # drop fragments left by the split
                out.append(sent)
    return out


def pack_passages(sentences, title, encoder):
    """Greedily pack sentences into <= MAX_TEXT_TOKEN passages. -> [(text, ntok)].

    The title prefix is reserved BEFORE packing, so nothing is ever cut mid-word.
    """
    title_tokens = len(encoder.encode(title + ": "))

    units = []                                     # sentence -> (text, ntok)
    for sent in sentences:
        ids = encoder.encode(sent)
        if len(ids) <= MAX_TEXT_TOKEN:
            units.append((sent, len(ids)))
        else:                                      # a single sentence over budget
            for i in range(0, len(ids), MAX_TEXT_TOKEN):
                piece = ids[i:i + MAX_TEXT_TOKEN]
                units.append((encoder.decode(piece), len(piece)))

    out, cur, cur_len = [], [], 0
    budget = MAX_TEXT_TOKEN - title_tokens          # the first passage carries the title
    for text, n in units:
        if cur and cur_len + n > budget:
            out.append((" ".join(cur), cur_len))
            cur, cur_len, budget = [], 0, MAX_TEXT_TOKEN
        cur.append(text)
        cur_len += n
    if cur:
        out.append((" ".join(cur), cur_len))

    if out:                                        # anchor the first passage
        head_text, head_len = out[0]
        out[0] = (title + ": " + head_text, head_len + title_tokens)
    return out


def main():
    encoder = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(DATASET, CONFIG, split="train")
    print(f"articles: {len(dataset):,}")

    order = list(range(len(dataset)))
    random.Random(SEED).shuffle(order)
    n_test = int(len(order) * TEST_RATIO)
    test_ids = set(order[:n_test])

    seen = set()
    rows = {"train": [], "test": []}
    tokens = {"train": 0, "test": 0}
    kept_articles = 0

    for i in order:
        item = dataset[i]
        passages = pack_passages(prose_sentences(item["text"]), item["title"], encoder)
        if not passages:
            continue
        kept_articles += 1
        split = "test" if i in test_ids else "train"
        for text, n in passages:
            if text in seen:                       # global de-duplication
                continue
            seen.add(text)
            rows[split].append({"text": text})
            tokens[split] += n

    os.makedirs(WIKI_DIR, exist_ok=True)
    for split, path in (("train", TRAIN_OUT), ("test", TEST_OUT)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows[split], f, ensure_ascii=False)
        print(f"{split:5s}: {len(rows[split]):>8,} passages  "
              f"{tokens[split] / 1e6:6.1f} M tokens  -> {path}")
    print(f"articles kept: {kept_articles:,}/{len(dataset):,}")

    # Output of the previous, broken version -- superseded by the two files above.
    for stale in ("wiki_pretrain_train.json", "wiki_pretrain_val.json"):
        path = os.path.join(WIKI_DIR, stale)
        if os.path.exists(path):
            os.remove(path)
            print(f"removed superseded {path}")


if __name__ == "__main__":
    main()
