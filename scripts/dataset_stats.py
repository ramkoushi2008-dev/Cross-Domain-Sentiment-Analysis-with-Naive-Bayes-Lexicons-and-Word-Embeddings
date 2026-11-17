"""Load and preprocess every domain and report its size, class balance and text length.

    python scripts/dataset_stats.py
Output: results/dataset_stats.csv

Text length matters because domains range from single financial sentences to long movie
reviews, which affects both accuracy and the domain distance measure.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import TARGET_DOMAINS, balanced_sample, load_imdb, train_val_split
from sentiment.preprocess import tokenize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def describe(name, role, raw, used):
    lengths = [len(tokenize(t)) for t, _ in used]
    vocab = {w for t, _ in used for w in tokenize(t)}
    return {
        "domain": name, "role": role,
        "raw_pos": sum(l for _, l in raw), "raw_neg": sum(1 - l for _, l in raw),
        "used": len(used), "used_pos": sum(l for _, l in used),
        "mean_tokens": round(statistics.mean(lengths), 1),
        "median_tokens": statistics.median(lengths),
        "vocab": len(vocab),
    }


def main():
    rows = []
    imdb_train = load_imdb("train")
    train, val = train_val_split(imdb_train)
    rows.append(describe("imdb_train", "train", imdb_train, train))
    rows.append(describe("imdb_val", "validation", imdb_train, val))
    imdb_test = load_imdb("test")
    rows.append(describe("imdb_test", "test (in-domain)", imdb_test, imdb_test))
    for name, loader in TARGET_DOMAINS.items():
        try:
            raw = loader()
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        rows.append(describe(name, "test (cross-domain)", raw, balanced_sample(raw)))

    cols = list(rows[0])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "dataset_stats.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"{'domain':<20}{'raw +/-':>16}{'used':>7}{'mean len':>10}{'vocab':>9}")
    for r in rows:
        print(f"{r['domain']:<20}{r['raw_pos']:>8}/{r['raw_neg']:<7}{r['used']:>7}"
              f"{r['mean_tokens']:>10}{r['vocab']:>9}")


if __name__ == "__main__":
    main()
