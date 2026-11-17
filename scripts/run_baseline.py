"""Reproduce the homework baseline on IMDB: multinomial NB, unigram counts, add-one smoothing.

Trains on the 20,000 IMDB training reviews left after holding out the 5,000-review
validation set, and reports validation and test metrics.

    python scripts/run_baseline.py
Outputs: results/baseline_imdb.json, results/baseline_top_features.csv
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import load_imdb, train_val_split
from sentiment.evaluate import format_report, report
from sentiment.features import featurize
from sentiment.naive_bayes import NaiveBayes

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    t0 = time.time()
    train, val = train_val_split(load_imdb("train"), n_val=5000, seed=42)
    test = load_imdb("test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    X_train = featurize([t for t, _ in train]); y_train = [l for _, l in train]
    X_val = featurize([t for t, _ in val]); y_val = [l for _, l in val]
    X_test = featurize([t for t, _ in test]); y_test = [l for _, l in test]
    print(f"tokenized in {time.time() - t0:.0f}s")

    nb = NaiveBayes(alpha=1.0).fit(X_train, y_train)
    results = {
        "model": "baseline: multinomial NB, unigram counts, add-one smoothing",
        "train_size": len(train),
        "vocab_size": len(nb.V),
        "validation": report(y_val, nb.predict(X_val)),
        "test": report(y_test, nb.predict(X_test)),
    }
    print(format_report("IMDB val ", results["validation"]))
    print(format_report("IMDB test", results["test"]))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "baseline_imdb.json"), "w") as f:
        json.dump(results, f, indent=2)

    pos, neg = nb.top_features(k=30, min_count=20)
    with open(os.path.join(RESULTS, "baseline_top_features.csv"), "w", encoding="utf-8") as f:
        f.write("rank,positive_word,log_ratio,negative_word,log_ratio\n")
        for i, (p, n) in enumerate(zip(pos, neg), 1):
            f.write(f"{i},{p},{nb.log_ratio(p):.3f},{n},{nb.log_ratio(n):.3f}\n")
    print("most positive:", ", ".join(pos[:10]))
    print("most negative:", ", ".join(neg[:10]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
