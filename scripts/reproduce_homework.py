"""Check that the ported classifier reproduces the Assignment 3 baseline exactly.

Assignment 3 trained multinomial NB (add-one smoothing, raw unigram counts, whitespace
tokens) on the Pang & Lee polarity dataset with a stratified 80/20 split (seed 42) and
reported accuracy 0.8375 with |V| = 45,621 (variant i). This script rebuilds that exact
split and setting with the project code, then also reports the project tokenizer on the
same split.

    python scripts/download_data.py review_polarity
    python scripts/reproduce_homework.py
Output: results/homework_reproduction.json
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import load_review_polarity
from sentiment.evaluate import format_report, report
from sentiment.naive_bayes import NaiveBayes
from sentiment.preprocess import tokenize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HOMEWORK = {"accuracy": 0.8375, "vocab_size": 45621}


def homework_split(docs, test_ratio=0.2, seed=42):
    """Identical to Assignment 3's train_test_split (same RNG call order)."""
    rng = random.Random(seed)
    train, test = [], []
    for label in (0, 1):
        group = [d for d in docs if d[1] == label]
        rng.shuffle(group)
        n_test = int(len(group) * test_ratio)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def run(train, test, tok):
    X_tr = [Counter(tok(t)) for t, _ in train]
    X_te = [Counter(tok(t)) for t, _ in test]
    nb = NaiveBayes(alpha=1.0).fit(X_tr, [l for _, l in train])
    return len(nb.V), report([l for _, l in test], nb.predict(X_te))


def main():
    train, test = homework_split(load_review_polarity())
    v_ws, r_ws = run(train, test, str.split)
    v_tok, r_tok = run(train, test, tokenize)
    matches = abs(r_ws["accuracy"] - HOMEWORK["accuracy"]) < 1e-9 and v_ws == HOMEWORK["vocab_size"]

    print(format_report("whitespace tokens ", r_ws), f"|V|={v_ws}")
    print(format_report("project tokenizer ", r_tok), f"|V|={v_tok}")
    print(f"Assignment 3 reported acc={HOMEWORK['accuracy']} |V|={HOMEWORK['vocab_size']} -> "
          f"{'EXACT MATCH' if matches else 'MISMATCH'}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "homework_reproduction.json"), "w") as f:
        json.dump({"dataset": "Pang & Lee polarity v2.0, 80/20 stratified split, seed 42",
                   "assignment3_reported": HOMEWORK,
                   "whitespace_tokens": {"vocab_size": v_ws, **r_ws},
                   "project_tokenizer": {"vocab_size": v_tok, **r_tok},
                   "exact_match": matches}, f, indent=2)
    sys.exit(0 if matches else 1)


if __name__ == "__main__":
    main()
