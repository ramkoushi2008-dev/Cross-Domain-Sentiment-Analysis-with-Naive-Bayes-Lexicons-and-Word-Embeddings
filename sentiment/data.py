"""Loaders for every domain, the IMDB validation split and balanced test sampling.

Every loader returns a list of (text, label) pairs with label 1 = positive, 0 = negative.
Raw files live in data/raw/ (created by scripts/download_data.py).
"""
import glob
import html
import os
import random
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

POS, NEG = 1, 0


# ---------------------------------------------------------------- IMDB (source domain)

def load_imdb(split):
    """IMDB Large Movie Review Dataset; split is 'train' or 'test' (25,000 each).

    Reads the original aclImdb folders, or the CSV written from the Hugging Face
    mirror (stanfordnlp/imdb, label 0 = neg, 1 = pos) when Stanford is unreachable.
    """
    csv = os.path.join(RAW, "imdb", f"{split}.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        return list(zip(df["text"].astype(str), df["label"].astype(int)))
    base = os.path.join(RAW, "imdb", "aclImdb", split)
    docs = []
    for name, label in (("pos", POS), ("neg", NEG)):
        for path in sorted(glob.glob(os.path.join(base, name, "*.txt"))):
            with open(path, encoding="utf-8") as f:
                docs.append((f.read(), label))
    if not docs:
        raise FileNotFoundError(f"No IMDB files under {base}; run scripts/download_data.py imdb")
    return docs


def load_review_polarity():
    """Pang & Lee (2004) polarity dataset v2.0: 1,000 pos + 1,000 neg pre-tokenized reviews.

    Used only to check that the ported classifier reproduces the Assignment 3 result.
    Order matches the homework loader (pos files, then neg, each sorted by name).
    """
    base = os.path.join(RAW, "review_polarity", "txt_sentoken")
    docs = []
    for name, label in (("pos", POS), ("neg", NEG)):
        for path in sorted(glob.glob(os.path.join(base, name, "*.txt"))):
            with open(path, encoding="utf-8") as f:
                docs.append((f.read(), label))
    return docs


def train_val_split(docs, n_val=5000, seed=42):
    """Stratified hold-out: n_val documents (half per class) become the validation set.

    The validation set is used for every tuning choice so the test sets stay untouched
    until final evaluation.
    """
    rng = random.Random(seed)
    train, val = [], []
    for label in (NEG, POS):
        group = [d for d in docs if d[1] == label]
        rng.shuffle(group)
        k = n_val // 2
        val.extend(group[:k])
        train.extend(group[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------- target domains

BLITZER_DOMAINS = {
    "amazon_books": "books",
    "amazon_dvd": "dvd",
    "amazon_electronics": "electronics",
    "amazon_kitchen": "kitchen_&_housewares",
}
_REVIEW_TEXT = re.compile(r"<review_text>(.*?)</review_text>", re.S)


def load_blitzer(domain):
    """Multi-Domain Sentiment Dataset (Blitzer et al., 2007): 1,000 pos + 1,000 neg per domain.

    The .review files are pseudo-XML; only the <review_text> field is used.
    """
    base = os.path.join(RAW, "blitzer", "sorted_data_acl", BLITZER_DOMAINS[domain])
    docs = []
    for fname, label in (("positive.review", POS), ("negative.review", NEG)):
        with open(os.path.join(base, fname), encoding="latin-1") as f:
            raw = f.read()
        docs.extend((html.unescape(t.strip()), label) for t in _REVIEW_TEXT.findall(raw))
    return docs


def load_yelp():
    """Yelp Review Polarity test split (Zhang et al., 2015); HF labels 0 = neg, 1 = pos."""
    df = pd.read_csv(os.path.join(RAW, "yelp", "test.csv"))
    return list(zip(df["text"].astype(str), df["label"].astype(int)))


def load_drugs():
    """Drugs.com reviews (Gräßer et al., 2018), test file.

    10-star ratings are mapped as in IMDB: >= 7 positive, <= 4 negative, 5-6 dropped.
    """
    df = pd.read_csv(os.path.join(RAW, "drugs", "drugsComTest_raw.tsv"), sep="\t")
    docs = []
    for text, rating in zip(df["review"].astype(str), df["rating"]):
        text = html.unescape(text).strip().strip('"')
        if rating >= 7:
            docs.append((text, POS))
        elif rating <= 4:
            docs.append((text, NEG))
    return docs


def load_finphrase(agreement="50Agree"):
    """Financial PhraseBank (Malo et al., 2014); neutral sentences are dropped."""
    path = os.path.join(RAW, "finphrase", "FinancialPhraseBank-v1.0", f"Sentences_{agreement}.txt")
    labels = {"positive": POS, "negative": NEG}
    docs = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            text, _, lab = line.strip().rpartition("@")
            if lab in labels:
                docs.append((text, labels[lab]))
    return docs


TARGET_DOMAINS = {
    **{name: (lambda n=name: load_blitzer(n)) for name in BLITZER_DOMAINS},
    "yelp": load_yelp,
    "drugs": load_drugs,
    "finphrase": load_finphrase,
}


def balanced_sample(docs, max_per_class=1000, seed=42):
    """Equal numbers of positive and negative texts, up to max_per_class each.

    If one class is smaller (e.g. 604 negative sentences in Financial PhraseBank),
    both classes are subsampled to that size.
    """
    rng = random.Random(seed)
    pos = [d for d in docs if d[1] == POS]
    neg = [d for d in docs if d[1] == NEG]
    n = min(len(pos), len(neg), max_per_class)
    sample = rng.sample(pos, n) + rng.sample(neg, n)
    rng.shuffle(sample)
    return sample


def load_target(domain, max_per_class=1000, seed=42):
    return balanced_sample(TARGET_DOMAINS[domain](), max_per_class, seed)
