"""Lexicon features on IMDB: baseline vs + Hu & Liu (vs + MPQA when the file is present).

Each model appends LEX_POS / LEX_NEG marker tokens to the bag of words; the classifier is
unchanged. Train = 20,000 IMDB reviews, validation = 5,000 held out, test = 25,000.

    python scripts/run_lexicon_features.py
Output: results/lexicon_features_imdb.json
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import load_imdb, train_val_split
from sentiment.evaluate import format_report, report
from sentiment.features import with_lexicon
from sentiment.lexicons import coverage, load_hu_liu, load_mpqa
from sentiment.naive_bayes import NaiveBayes
from sentiment.preprocess import tokenize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    t0 = time.time()
    train, val = train_val_split(load_imdb("train"), n_val=5000, seed=42)
    test = load_imdb("test")
    # tokenize once; every model only adds marker tokens on top
    T = {name: [tokenize(t) for t, _ in split] for name, split in
         (("train", train), ("val", val), ("test", test))}
    Y = {"train": [l for _, l in train], "val": [l for _, l in val], "test": [l for _, l in test]}
    print(f"tokenized in {time.time() - t0:.0f}s")

    lexicons = {"hu_liu": load_hu_liu()}
    try:
        lexicons["mpqa"] = load_mpqa()
    except FileNotFoundError as e:
        print("MPQA skipped:", str(e).split(".")[0])

    models = {"baseline": None}
    models.update({f"+{name}": with_lexicon(lex) for name, lex in lexicons.items()})

    out = {"lexicons": {}, "models": {}}
    for name, lex in lexicons.items():
        out["lexicons"][name] = {
            "size": len(lex),
            "positive": sum(v == 1 for v in lex.values()),
            "negative": sum(v == -1 for v in lex.values()),
            "imdb_train_coverage": coverage(lex, T["train"]),
        }

    for name, transform in models.items():
        feats = {k: [Counter(transform(t) if transform else t) for t in toks] for k, toks in T.items()}
        nb = NaiveBayes(alpha=1.0).fit(feats["train"], Y["train"])
        res = {"validation": report(Y["val"], nb.predict(feats["val"])),
               "test": report(Y["test"], nb.predict(feats["test"]))}
        if transform:
            # learned weight of the marker tokens: log P(LEX_x | pos) - log P(LEX_x | neg)
            res["marker_log_ratio"] = {m: nb.log_ratio(m) for m in ("LEX_POS", "LEX_NEG") if m in nb.V}
        out["models"][name] = res
        print(format_report(f"{name:<9} val ", res["validation"]))
        print(format_report(f"{name:<9} test", res["test"]))

    base = out["models"]["baseline"]
    for name, res in out["models"].items():
        res["delta_vs_baseline"] = {s: res[s]["accuracy"] - base[s]["accuracy"] for s in ("validation", "test")}

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "lexicon_features_imdb.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v["delta_vs_baseline"] for k, v in out["models"].items()}, indent=1))
    print(json.dumps(out["lexicons"], indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
