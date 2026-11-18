"""Negation handling on IMDB: three methods compared (proposal, method 3d).

  1. NOT_ prefix on every word from a negator to the next punctuation mark
  2. NOT_ prefix on the next k words only (k tuned on validation)
  3. Flipping Hu & Liu lexicon markers inside the negation scope ("not good" -> LEX_NEG);
     tested on top of the Hu & Liu model, since it changes only the lexicon tokens.
     Its scope (to punctuation, or the best k) is also chosen on validation.

    python scripts/run_negation.py
Output: results/negation_imdb.json, results/negation_k_tuning.csv
"""
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import load_imdb, train_val_split
from sentiment.evaluate import format_report, report
from sentiment.features import negate_words, negation_scope, with_flipped_lexicon, with_lexicon
from sentiment.lexicons import load_hu_liu
from sentiment.naive_bayes import NaiveBayes
from sentiment.preprocess import tokenize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10]


def fit_eval(T, Y, transform, splits=("val",)):
    feats = {s: [Counter(transform(t) if transform else t) for t in T[s]] for s in ("train", *splits)}
    nb = NaiveBayes(alpha=1.0).fit(feats["train"], Y["train"])
    return nb, {s: report(Y[s], nb.predict(feats[s])) for s in splits}


def scope_stats(token_lists):
    """How much text negation touches: share of tokens in scope, scope lengths."""
    in_scope = total = docs_with = 0
    lengths = []
    for toks in token_lists:
        mask = negation_scope(toks)
        in_scope += sum(mask)
        total += len(toks)
        docs_with += any(mask)
        run = 0
        for m in mask + [False]:
            if m:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
    return {"token_share_in_scope": in_scope / total, "docs_with_negation": docs_with / len(token_lists),
            "n_scopes": len(lengths), "mean_scope_len": statistics.mean(lengths),
            "median_scope_len": statistics.median(lengths)}


def main():
    t0 = time.time()
    train, val = train_val_split(load_imdb("train"), n_val=5000, seed=42)
    test = load_imdb("test")
    T = {n: [tokenize(t) for t, _ in d] for n, d in (("train", train), ("val", val), ("test", test))}
    Y = {n: [l for _, l in d] for n, d in (("train", train), ("val", val), ("test", test))}
    hl = load_hu_liu()
    stats = scope_stats(T["train"])
    print(f"loaded in {time.time() - t0:.0f}s; scope stats: {stats}")

    # tune k for method 2 on validation
    tuning = []
    _, r = fit_eval(T, Y, negate_words(None))
    tuning.append({"method": "NOT_to_punct", "k": "punct", "val_accuracy": r["val"]["accuracy"]})
    for k in K_VALUES:
        _, r = fit_eval(T, Y, negate_words(k))
        tuning.append({"method": "NOT_next_k", "k": k, "val_accuracy": r["val"]["accuracy"]})
        print(f"NOT_ next k={k:<2} val={r['val']['accuracy']:.4f}")
    best_k = max((t for t in tuning if t["method"] == "NOT_next_k"), key=lambda t: t["val_accuracy"])["k"]

    # choose the flipping scope on validation
    for k in (None, best_k):
        _, r = fit_eval(T, Y, with_flipped_lexicon(hl, k=k))
        tuning.append({"method": "flip_lexicon", "k": "punct" if k is None else k,
                       "val_accuracy": r["val"]["accuracy"]})
    flips = [t for t in tuning if t["method"] == "flip_lexicon"]
    best_flip = max(flips, key=lambda t: t["val_accuracy"])
    flip_k = None if best_flip["k"] == "punct" else best_flip["k"]
    for t in tuning:
        if t["method"] != "NOT_next_k":
            print(f"{t['method']:<14} scope={t['k']!s:<6} val={t['val_accuracy']:.4f}")

    # final comparison on validation + test
    configs = {
        "baseline": None,
        "NOT_to_punct": negate_words(None),
        f"NOT_next_k{best_k}": negate_words(best_k),
        "+hu_liu": with_lexicon(hl),
        f"+hu_liu_flipped({best_flip['k']})": with_flipped_lexicon(hl, k=flip_k),
    }
    final = {}
    for name, tr in configs.items():
        nb, res = fit_eval(T, Y, tr, splits=("val", "test"))
        final[name] = {"validation": res["val"], "test": res["test"]}
        if "hu_liu" in name:
            final[name]["marker_log_ratio"] = {m: nb.log_ratio(m) for m in ("LEX_POS", "LEX_NEG")}
        if name.startswith("NOT_"):
            pos, neg = nb.top_features(k=400, min_count=50)
            final[name]["top_NOT_features"] = {
                "positive": [w for w in pos if w.startswith("NOT_")][:10],
                "negative": [w for w in neg if w.startswith("NOT_")][:10]}
        print(format_report(f"{name:<24} test", res["test"]))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "negation_k_tuning.csv"), "w", newline="\n") as f:
        f.write("method,scope,val_accuracy\n")
        for t in tuning:
            f.write(f"{t['method']},{t['k']},{t['val_accuracy']:.4f}\n")
    with open(os.path.join(RESULTS, "negation_imdb.json"), "w", newline="\n") as f:
        json.dump({"scope_stats_imdb_train": stats, "best_k": best_k, "flip_scope": best_flip["k"],
                   "tuning": tuning, "final": final}, f, indent=2)
    for name, v in final.items():
        print(f"{name:<26} val={v['validation']['accuracy']:.4f} test={v['test']['accuracy']:.4f} "
              f"delta_test={v['test']['accuracy'] - final['baseline']['test']['accuracy']:+.4f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
