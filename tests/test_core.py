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


def test_tokenize_edge_cases():
    # real tags removed, emoticons kept
    assert tokenize("see <i>Boomerang</i>.<hr>Yuck >.< </3") == \
        ["see", "boomerang", ".", "yuck", ">", ".", "<", "<", "/", "3"]
    # accented letters stay inside words; decimals stay whole
    assert tokenize("Café Señor costs $9.99") == ["café", "señor", "costs", "$", "9.99"]
    # Windows-1252 dashes/quotes and typographic apostrophes normalised
    assert tokenize("good\x96really \x93fun\x94 shouldn’t") == \
        ["good", "-", "really", '"', "fun", '"', "should", "n't"]
    assert tokenize("well-made, must_see") == ["well-made", ",", "must", "_", "see"]


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


def test_lexicon_markers_appended_not_replacing():
    from sentiment.features import lexicon_markers, with_lexicon
    lex = {"good": 1, "boring": -1, "great": 1}
    toks = ["a", "good", "but", "boring", "film", ",", "good"]
    assert lexicon_markers(toks, lex) == ["LEX_POS", "LEX_NEG", "LEX_POS"]
    assert lexicon_markers(toks, lex, name="HL") == ["HL_POS", "HL_NEG", "HL_POS"]
    assert with_lexicon(lex)(toks) == toks + ["LEX_POS", "LEX_NEG", "LEX_POS"]


def test_hu_liu_lexicon_loads():
    from sentiment.lexicons import HU_LIU_DIR, load_hu_liu
    if not os.path.isdir(HU_LIU_DIR):
        return  # data not downloaded
    lex = load_hu_liu()
    assert 6700 < len(lex) < 6800
    assert lex["good"] == 1 and lex["bad"] == -1 and lex["naïve"] == -1


def test_parse_mpqa_line():
    from sentiment.lexicons import parse_mpqa_line
    e = parse_mpqa_line("type=strongsubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative")
    assert e["word1"] == "abandoned" and e["priorpolarity"] == "negative"


class _FakeKV:
    """Tiny stand-in for gensim KeyedVectors with 2-d vectors."""
    def __init__(self, vecs):
        import numpy as np
        self.vecs = {w: np.array(v, dtype=float) for w, v in vecs.items()}
        self.key_to_index = {w: i for i, w in enumerate(vecs)}

    def __getitem__(self, w):
        return self.vecs[w]


def test_expansion_gaps_margin_and_seed_priority():
    from sentiment.expansion import compare_labels, expand, label_by_margin, polarity_gaps
    kv = _FakeKV({"good": [1, 0], "great": [0.9, 0.1], "bad": [0, 1], "awful": [0.1, 0.9],
                  "superb": [0.95, 0.05], "dreadful": [0.05, 0.95], "meh": [0.7, 0.7]})
    seed = {"good": 1, "great": 1, "bad": -1, "awful": -1}
    gaps = polarity_gaps(["superb", "dreadful", "meh"], seed, kv, k=2)
    assert gaps["superb"] > 0.5 and gaps["dreadful"] < -0.5 and abs(gaps["meh"]) < 1e-6
    assert label_by_margin(gaps, 0.1) == {"superb": 1, "dreadful": -1}  # "meh" left out
    # seed labels always win over embedding labels
    assert expand(seed, {"good": -0.9, "superb": 0.8}, 0.1) == {**seed, "superb": 1}
    rows = compare_labels({"superb": 0.8, "dreadful": 0.2}, {"superb": 1, "dreadful": -1}, [0.0, 0.5])
    assert rows[0]["accuracy"] == 0.5 and rows[1]["labelled"] == 1 and rows[1]["accuracy"] == 1.0


def test_candidate_words_filters():
    from sentiment.expansion import candidate_words
    kv = _FakeKV({"plot": [1, 0], "good": [1, 0], "rare": [0, 1], "well-made": [1, 1], "!": [0, 1]})
    counts = {"plot": 50, "good": 90, "rare": 3, "well-made": 25, "!": 900, "oov": 99}
    assert candidate_words(counts, {"good": 1}, kv, min_count=20) == ["plot", "well-made"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
