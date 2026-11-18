"""Expand a polarity lexicon with GloVe embeddings (proposal, method 3c).

For each candidate word (an IMDB training word seen at least 20 times that is missing
from the seed lexicon), average its cosine similarity to its k = 10 nearest positive and
k = 10 nearest negative seed words. The difference

    gap(w) = mean_sim(w, 10 nearest positive) - mean_sim(w, 10 nearest negative)

decides the label: positive if gap > margin, negative if gap < -margin, otherwise the word
is left out. The margin is tuned on the IMDB validation set.

Known weakness: standard embeddings put antonyms ("good"/"bad") close together, because
they occur in the same contexts (Maas et al., 2011), so the gap is often small. Label
quality is therefore checked on words whose polarity is known (held-out seed words now,
MPQA labels once that lexicon is available).
"""
import os
import random
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENSIM_DIR = os.path.join(ROOT, "data", "raw", "gensim-data")
KV_CACHE = os.path.join(GENSIM_DIR, "glove-wiki-gigaword-100.kv")
_WORD = re.compile(r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$")


def load_glove():
    """GloVe 6B 100d (Wikipedia + Gigaword) as gensim KeyedVectors, cached after first load."""
    from gensim.models import KeyedVectors
    if os.path.exists(KV_CACHE):
        return KeyedVectors.load(KV_CACHE, mmap="r")
    os.environ.setdefault("GENSIM_DATA_DIR", GENSIM_DIR)
    import gensim.downloader as api
    kv = api.load("glove-wiki-gigaword-100")
    kv.save(KV_CACHE)
    return kv


def candidate_words(token_counts, seed_lexicon, kv, min_count=20):
    """Words seen >= min_count times, alphabetic, not in the seed lexicon, present in GloVe."""
    return sorted(w for w, n in token_counts.items()
                  if n >= min_count and w not in seed_lexicon and _WORD.match(w)
                  and w in kv.key_to_index)


def _unit_rows(kv, words):
    m = np.stack([kv[w] for w in words]).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _mean_top_k(sims, k):
    """Mean of the k largest values in each row."""
    k = min(k, sims.shape[1])
    return np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)


def polarity_gaps(words, seed_lexicon, kv, k=10):
    """gap(w) for every word, using the seed words that have a GloVe vector."""
    pos = [w for w, p in seed_lexicon.items() if p == 1 and w in kv.key_to_index]
    neg = [w for w, p in seed_lexicon.items() if p == -1 and w in kv.key_to_index]
    W, P, N = _unit_rows(kv, words), _unit_rows(kv, pos), _unit_rows(kv, neg)
    gaps = _mean_top_k(W @ P.T, k) - _mean_top_k(W @ N.T, k)
    return dict(zip(words, gaps.tolist()))


def label_by_margin(gaps, margin):
    """Words whose |gap| exceeds the margin, labelled +1 / -1."""
    return {w: (1 if g > 0 else -1) for w, g in gaps.items() if abs(g) > margin}


def expand(seed_lexicon, gaps, margin):
    """Seed lexicon plus the newly labelled words (seed labels always win)."""
    out = dict(label_by_margin(gaps, margin))
    out.update(seed_lexicon)
    return out


def heldout_label_check(seed_lexicon, kv, margins, k=10, frac=0.1, seed=42):
    """Hide a stratified 10% of the seed words, label them from the other 90%, and report
    accuracy and coverage at each margin. Tests whether embedding similarity recovers
    known polarities (the antonym problem shows up as low accuracy at small margins)."""
    rng = random.Random(seed)
    known = [w for w in seed_lexicon if w in kv.key_to_index]
    held = set()
    for pol in (1, -1):
        group = sorted(w for w in known if seed_lexicon[w] == pol)
        held |= set(rng.sample(group, int(len(group) * frac)))
    train_lex = {w: p for w, p in seed_lexicon.items() if w not in held}
    gaps = polarity_gaps(sorted(held), train_lex, kv, k)
    return compare_labels(gaps, seed_lexicon, margins)


def compare_labels(gaps, gold, margins):
    """Accuracy and coverage of margin-thresholded labels against a gold lexicon."""
    shared = [w for w in gaps if w in gold]
    rows = []
    for m in margins:
        labelled = [w for w in shared if abs(gaps[w]) > m]
        correct = sum((gaps[w] > 0) == (gold[w] == 1) for w in labelled)
        rows.append({"margin": m, "n_gold": len(shared), "labelled": len(labelled),
                     "coverage": len(labelled) / len(shared) if shared else 0.0,
                     "accuracy": correct / len(labelled) if labelled else None})
    return rows
