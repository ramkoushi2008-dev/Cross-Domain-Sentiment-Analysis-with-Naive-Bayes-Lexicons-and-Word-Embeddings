"""Feature extraction. Week 1: plain bag-of-words unigram counts (the baseline).

Later feature designs (lexicon tokens, NOT_ marking) append marker tokens to the token
list before counting, so they all plug into the same `featurize` call.
"""
from collections import Counter

from .preprocess import tokenize


def bag_of_words(tokens):
    return Counter(tokens)


def featurize(texts, transform=None):
    """Tokenize each text, optionally transform the token list, and count."""
    out = []
    for t in texts:
        toks = tokenize(t)
        if transform is not None:
            toks = transform(toks)
        out.append(Counter(toks))
    return out
