"""Unit tests for preprocessing, the data split, the classifier and the metrics.

Run with:  python -m pytest tests   (or: python tests/test_core.py)
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.data import balanced_sample, train_val_split
from sentiment.evaluate import report
from sentiment.naive_bayes import NaiveBayes
from sentiment.preprocess import tokenize


def test_tokenize_keeps_negation_and_strips_html():
    assert tokenize("Not good.<br /><br />I didn't LIKE it") == \
        ["not", "good", ".", "i", "did", "n't", "like", "it"]
    assert tokenize("can't won't") == ["ca", "n't", "wo", "n't"]
    assert tokenize("a &amp; b") == ["a", "&", "b"]


def test_train_val_split_is_stratified_and_disjoint():
    docs = [(f"d{i}", i % 2) for i in range(100)]
    train, val = train_val_split(docs, n_val=20, seed=0)
    assert len(val) == 20 and len(train) == 80
    assert sum(l for _, l in val) == 10
    assert not set(t for t, _ in train) & set(t for t, _ in val)
    assert train_val_split(docs, 20, 0) == (train, val)  # deterministic


def test_balanced_sample_caps_at_smaller_class():
    docs = [("p", 1)] * 50 + [("n", 0)] * 8
    s = balanced_sample(docs, max_per_class=1000)
    assert len(s) == 16 and sum(l for _, l in s) == 8


def test_naive_bayes_matches_hand_computation():
    # Jurafsky & Martin (3rd ed.) worked example, section 4.3
    train = [("just plain boring", 0), ("entirely predictable and lacks energy", 0),
             ("no surprises and very few laughs", 0), ("very powerful", 1),
             ("the most fun film of the summer", 1)]
    X = [Counter(t.split()) for t, _ in train]
    y = [l for _, l in train]
    nb = NaiveBayes(alpha=1.0).fit(X, y)
    test = Counter("predictable with no fun".split())
    sc = nb.scores(test)
    # P(-)P(S|-) = 3/5 * (2*2*1)/34^3 ; P(+)P(S|+) = 2/5 * (1*1*2)/29^3
    import math
    assert math.isclose(sc[0], math.log(3 / 5 * 4 / 34 ** 3))
    assert math.isclose(sc[1], math.log(2 / 5 * 2 / 29 ** 3))
    assert nb.predict_one(test) == 0


def test_report_matches_sklearn():
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    r = report(y_true, y_pred)
    p, rc, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    assert abs(r["accuracy"] - accuracy_score(y_true, y_pred)) < 1e-12
    assert abs(r["pos"]["precision"] - p[1]) < 1e-12 and abs(r["neg"]["recall"] - rc[0]) < 1e-12
    assert abs(r["macro_f1"] - f.mean()) < 1e-12
    assert r["confusion"] == confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
