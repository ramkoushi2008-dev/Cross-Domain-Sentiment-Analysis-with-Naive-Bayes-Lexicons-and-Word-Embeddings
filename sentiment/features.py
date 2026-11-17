"""Feature extraction.

Every feature design appends marker tokens to (or rewrites) the token list before
counting, so the Naive Bayes classifier itself never changes between experiments.

- bag of words (baseline): unigram counts
- lexicon features: one LEX_POS / LEX_NEG token per lexicon word in the review
"""
from collections import Counter

from .preprocess import tokenize


def bag_of_words(tokens):
    return Counter(tokens)


def lexicon_markers(tokens, lexicon, name="LEX"):
    """Marker tokens for every lexicon word: e.g. ['good', 'film'] -> ['LEX_POS'].

    `name` keeps the markers of different lexicons apart when they are combined
    (e.g. HL_POS vs MPQA_POS); a single-lexicon model uses LEX_POS / LEX_NEG.
    Marker names are upper-case, so they can never collide with the lower-cased words.
    """
    out = []
    for t in tokens:
        pol = lexicon.get(t)
        if pol == 1:
            out.append(f"{name}_POS")
        elif pol == -1:
            out.append(f"{name}_NEG")
    return out


def with_lexicon(lexicon, name="LEX"):
    """Token transform: keep the words and append the lexicon markers."""
    def transform(tokens):
        return tokens + lexicon_markers(tokens, lexicon, name)
    return transform


def featurize(texts, transform=None):
    """Tokenize each text, optionally transform the token list, and count."""
    out = []
    for t in texts:
        toks = tokenize(t)
        if transform is not None:
            toks = transform(toks)
        out.append(Counter(toks))
    return out
