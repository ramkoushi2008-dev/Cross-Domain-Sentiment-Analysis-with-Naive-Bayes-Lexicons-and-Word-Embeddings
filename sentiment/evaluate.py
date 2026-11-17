"""Evaluation metrics: accuracy, per-class precision/recall/F1 and the confusion matrix.

Implemented directly (rather than via scikit-learn) so the numbers can be checked by hand;
tests/test_core.py cross-checks them against scikit-learn.
"""


def confusion(y_true, y_pred, pos=1, neg=0):
    """Return {'tp','fp','tn','fn'} with `pos` as the positive class."""
    m = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for t, p in zip(y_true, y_pred):
        if p == pos:
            m["tp" if t == pos else "fp"] += 1
        else:
            m["tn" if t == neg else "fn"] += 1
    return m


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report(y_true, y_pred):
    """All metrics for one model on one test set, as a JSON-friendly dict."""
    c = confusion(y_true, y_pred)
    p_pos, r_pos, f_pos = _prf(c["tp"], c["fp"], c["fn"])
    p_neg, r_neg, f_neg = _prf(c["tn"], c["fn"], c["fp"])
    n = len(y_true)
    return {
        "n": n,
        "accuracy": (c["tp"] + c["tn"]) / n,
        "pos": {"precision": p_pos, "recall": r_pos, "f1": f_pos},
        "neg": {"precision": p_neg, "recall": r_neg, "f1": f_neg},
        "macro_f1": (f_pos + f_neg) / 2,
        # rows = gold (neg, pos), columns = predicted (neg, pos)
        "confusion": [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
    }


def format_report(name, r):
    (tn, fp), (fn, tp) = r["confusion"]
    return (f"{name}: acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
            f"P/R pos={r['pos']['precision']:.3f}/{r['pos']['recall']:.3f}  "
            f"neg={r['neg']['precision']:.3f}/{r['neg']['recall']:.3f}  "
            f"[[TN {tn} FP {fp}] [FN {fn} TP {tp}]]")
