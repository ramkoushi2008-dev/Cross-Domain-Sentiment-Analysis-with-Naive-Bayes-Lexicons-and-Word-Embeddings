"""GloVe-expanded Hu & Liu lexicon: tune the margin on IMDB validation, then test once.

Two scoring variants are tuned:
  raw      - gap(w) exactly as in the proposal
  centered - gap(w) minus the median gap over all candidates. Added as a check for
             polarity bias: the negative seed list is 2.4x larger, and the raw labels are
             skewed negative at larger margins (e.g. 863 negative vs 400 positive words at
             margin 0.10), although the median raw gap itself is close to zero (-0.008).
The (variant, margin) with the best validation accuracy is evaluated on the test set,
together with the best raw configuration (the method as proposed).

    python scripts/run_expansion.py
Outputs: results/expansion_tuning.csv, results/expansion_imdb.json, results/expanded_lexicon.csv
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
from sentiment.expansion import (candidate_words, expand, heldout_label_check, label_by_margin,
                                 load_glove, polarity_gaps)
from sentiment.features import with_lexicon
from sentiment.lexicons import load_hu_liu
from sentiment.naive_bayes import NaiveBayes
from sentiment.preprocess import tokenize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MARGINS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]


def fit_eval(T, Y, transform, splits=("val",)):
    feats = {s: [Counter(transform(t) if transform else t) for t in T[s]] for s in ("train", *splits)}
    nb = NaiveBayes(alpha=1.0).fit(feats["train"], Y["train"])
    return nb, {s: report(Y[s], nb.predict(feats[s])) for s in splits}


def main():
    t0 = time.time()
    train, val = train_val_split(load_imdb("train"), n_val=5000, seed=42)
    test = load_imdb("test")
    T = {n: [tokenize(t) for t, _ in d] for n, d in (("train", train), ("val", val), ("test", test))}
    Y = {n: [l for _, l in d] for n, d in (("train", train), ("val", val), ("test", test))}
    hl = load_hu_liu()
    kv = load_glove()
    print(f"data + GloVe loaded in {time.time() - t0:.0f}s")

    # 1. candidate words and their polarity gaps (training data only)
    counts = Counter(w for toks in T["train"] for w in toks)
    cands = candidate_words(counts, hl, kv, min_count=20)
    raw = polarity_gaps(cands, hl, kv, k=10)
    med = statistics.median(raw.values())
    gaps = {"raw": raw, "centered": {w: g - med for w, g in raw.items()}}
    print(f"{len(cands)} candidate words; median raw gap {med:+.3f}")

    # 2. label quality on held-out Hu & Liu words (MPQA comparison once that file is available)
    heldout = heldout_label_check(hl, kv, MARGINS)

    # 3. tune (variant, margin) on validation
    rows = []
    for variant, g in gaps.items():
        for m in MARGINS:
            added = label_by_margin(g, m)
            _, r = fit_eval(T, Y, with_lexicon(expand(hl, g, m)))
            rows.append({"variant": variant, "margin": m, "added": len(added),
                         "added_pos": sum(v == 1 for v in added.values()),
                         "added_neg": sum(v == -1 for v in added.values()),
                         "val_accuracy": r["val"]["accuracy"]})
            print(f"{variant:<8} m={m:.2f} added={len(added):5d} "
                  f"(+{rows[-1]['added_pos']}/-{rows[-1]['added_neg']}) val={r['val']['accuracy']:.4f}")
    best = max(rows, key=lambda r: r["val_accuracy"])
    best_raw = max((r for r in rows if r["variant"] == "raw"), key=lambda r: r["val_accuracy"])

    # 4. final evaluation on test: baseline, + Hu & Liu, + expanded (best raw, best overall)
    final = {}
    configs = {"baseline": None, "+hu_liu": with_lexicon(hl)}
    for r in {id(best_raw): best_raw, id(best): best}.values():
        name = f"+expanded_{r['variant']}_m{r['margin']:.2f}"
        configs[name] = with_lexicon(expand(hl, gaps[r["variant"]], r["margin"]))
    for name, tr in configs.items():
        nb, res = fit_eval(T, Y, tr, splits=("val", "test"))
        final[name] = {"validation": res["val"], "test": res["test"]}
        if tr:
            final[name]["marker_log_ratio"] = {m: nb.log_ratio(m) for m in ("LEX_POS", "LEX_NEG")}
        print(format_report(f"{name:<26} test", res["test"]))

    # 5. save
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "expansion_tuning.csv"), "w", newline="\n") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    chosen = gaps[best["variant"]]
    added = label_by_margin(chosen, best["margin"])
    with open(os.path.join(RESULTS, "expanded_lexicon.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("word,label,gap,imdb_train_count\n")
        for w in sorted(added, key=lambda w: -abs(chosen[w])):
            f.write(f"{w},{'positive' if added[w] == 1 else 'negative'},{chosen[w]:.4f},{counts[w]}\n")
    top_pos = sorted((w for w in added if added[w] == 1), key=lambda w: -chosen[w])[:15]
    top_neg = sorted((w for w in added if added[w] == -1), key=lambda w: chosen[w])[:15]
    out = {"n_candidates": len(cands), "median_raw_gap": med, "heldout_label_check": heldout,
           "best_validation": best, "best_raw_validation": best_raw,
           "examples_added": {"positive": top_pos, "negative": top_neg}, "final": final}
    with open(os.path.join(RESULTS, "expansion_imdb.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=2)
    print("best:", best, "\nbest raw:", best_raw)
    print("added +:", ", ".join(top_pos), "\nadded -:", ", ".join(top_neg))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
