"""IMDB feature ablation and cross-domain tests (proposal, method 4).

Every model is trained once on the 20,000 IMDB training reviews and tested on the IMDB
validation and test sets and on each target domain (balanced, up to 1,000 per class).
Features are added one at a time and then all together. All settings come from earlier
validation tuning: expansion margin 0.15 (results/expanded_lexicon.csv), negation k = 2.

The MPQA lexicon is not yet available, so the combined model uses Hu & Liu + its GloVe
expansion; MPQA will be added when the lexicon file arrives.

    python scripts/run_ablation_crossdomain.py
Outputs: results/ablation_crossdomain.json, results/accuracy_by_domain.csv,
         results/delta_by_domain.csv
"""
import csv
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import TARGET_DOMAINS, load_imdb, load_target, train_val_split
from sentiment.evaluate import report
from sentiment.features import combined, negate_words, with_flipped_lexicon, with_lexicon
from sentiment.lexicons import coverage, load_hu_liu
from sentiment.naive_bayes import NaiveBayes
from sentiment.preprocess import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
K = 2


def load_expanded(hl):
    lex = dict(hl)
    with open(os.path.join(RESULTS, "expanded_lexicon.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lex.setdefault(row["word"], 1 if row["label"] == "positive" else -1)
    return lex


def main():
    t0 = time.time()
    hl = load_hu_liu()
    exp = load_expanded(hl)
    models = {
        "baseline": None,
        "+hu_liu": with_lexicon(hl),
        "+expanded": with_lexicon(exp),
        f"NOT_k{K}": negate_words(K),
        f"+hu_liu_flip_k{K}": with_flipped_lexicon(hl, k=K),
        "combined": combined(exp, k=K),
    }

    train, val = train_val_split(load_imdb("train"), n_val=5000, seed=42)
    sets = {"imdb_val": val, "imdb_test": load_imdb("test")}
    for d in TARGET_DOMAINS:
        try:
            sets[d] = load_target(d)
        except FileNotFoundError:
            print(f"skip {d}: data not available")
    T_train = [tokenize(t) for t, _ in train]
    y_train = [l for _, l in train]
    T = {s: [tokenize(t) for t, _ in docs] for s, docs in sets.items()}
    Y = {s: [l for _, l in docs] for s, docs in sets.items()}
    print(f"loaded {len(sets)} test sets in {time.time() - t0:.0f}s")

    # domain diagnostics: unseen-word rate and lexicon coverage
    train_vocab = {w for toks in T_train for w in toks}
    diag = {}
    for s, toks_list in T.items():
        n = sum(len(t) for t in toks_list)
        oov = sum(1 for t in toks_list for w in t if w not in train_vocab)
        diag[s] = {"n_docs": len(toks_list), "oov_token_rate": oov / n,
                   "hu_liu_token_coverage": coverage(hl, toks_list)["token_coverage"]}

    results = {}
    for name, tr in models.items():
        nb = NaiveBayes(alpha=1.0).fit([Counter(tr(t) if tr else t) for t in T_train], y_train)
        results[name] = {}
        for s, toks_list in T.items():
            pred = nb.predict([Counter(tr(t) if tr else t) for t in toks_list])
            results[name][s] = report(Y[s], pred)
        print(f"{name:<16}" + " ".join(f"{s}={results[name][s]['accuracy']:.3f}" for s in T))

    # tables: accuracy, and change vs baseline (percentage points)
    sets_order = list(T)
    with open(os.path.join(RESULTS, "accuracy_by_domain.csv"), "w", newline="\n") as f:
        f.write("model," + ",".join(sets_order) + ",mean_target_domains\n")
        for name, r in results.items():
            tgt = [r[s]["accuracy"] for s in sets_order if not s.startswith("imdb")]
            f.write(name + "," + ",".join(f"{r[s]['accuracy']:.4f}" for s in sets_order)
                    + f",{sum(tgt) / len(tgt):.4f}\n")
    base = results["baseline"]
    with open(os.path.join(RESULTS, "delta_by_domain.csv"), "w", newline="\n") as f:
        f.write("model," + ",".join(sets_order) + "\n")
        for name, r in results.items():
            f.write(name + "," + ",".join(
                f"{100 * (r[s]['accuracy'] - base[s]['accuracy']):+.2f}" for s in sets_order) + "\n")
    with open(os.path.join(RESULTS, "ablation_crossdomain.json"), "w", newline="\n") as f:
        json.dump({"settings": {"negation_k": K, "expansion_margin": 0.15,
                                "expanded_lexicon_size": len(exp), "hu_liu_size": len(hl)},
                   "domain_diagnostics": diag, "results": results}, f, indent=2)
    print(json.dumps(diag, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
