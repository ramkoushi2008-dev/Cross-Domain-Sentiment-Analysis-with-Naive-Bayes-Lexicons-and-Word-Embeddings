"""Feature extraction.

Every feature design appends marker tokens to (or rewrites) the token list before
counting, so the Naive Bayes classifier itself never changes between experiments.

- bag of words (baseline): unigram counts
- lexicon features: one LEX_POS / LEX_NEG token per lexicon word in the review
- negation: NOT_ prefix to the next punctuation mark, NOT_ prefix on the next k words,
  or flipping lexicon markers inside the negation scope
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


# ------------------------------------------------------------------ negation

# Negation cues: the proposal's "not", "never", "n't", plus "no" and "cannot", which the
# tokenizer keeps as single tokens ("no plot", "cannot recommend").
NEGATORS = {"not", "no", "never", "n't", "cannot"}
# Clause-ending punctuation closes a negation scope (Das & Chen, 2001; Pang et al., 2002).
SCOPE_END = {".", ",", ";", ":", "!", "?"}


def negation_scope(tokens, k=None):
    """Boolean mask: True for tokens inside a negation scope.

    A scope opens after a negator and closes at clause punctuation. With k set, it also
    closes after k words. The negator itself is not in its own scope; a second negator
    inside a scope starts a fresh scope (double negation is not modelled).
    """
    mask, remaining = [], 0
    for t in tokens:
        if t in SCOPE_END:
            remaining = 0
            mask.append(False)
        elif t in NEGATORS:
            remaining = k if k is not None else float("inf")
            mask.append(False)
        else:
            mask.append(remaining > 0)
            if remaining > 0:
                remaining -= 1
    return mask


def negate_words(k=None):
    """Token transform: prefix NOT_ to every word in a negation scope.

    k=None -> scope runs to the next punctuation mark (method 1);
    k=int  -> only the next k words, still stopping at punctuation (method 2).
    """
    def transform(tokens):
        return [("NOT_" + t if neg else t) for t, neg in zip(tokens, negation_scope(tokens, k))]
    return transform


def lexicon_markers_flipped(tokens, lexicon, name="LEX", k=None):
    """Lexicon markers whose polarity is flipped inside a negation scope (method 3):
    "not good" -> LEX_NEG instead of LEX_POS. The words themselves are unchanged."""
    out = []
    for t, neg in zip(tokens, negation_scope(tokens, k)):
        pol = lexicon.get(t)
        if pol is None:
            continue
        if neg:
            pol = -pol
        out.append(f"{name}_POS" if pol == 1 else f"{name}_NEG")
    return out


def with_flipped_lexicon(lexicon, name="LEX", k=None):
    def transform(tokens):
        return tokens + lexicon_markers_flipped(tokens, lexicon, name, k)
    return transform


def compose(*transforms):
    """Apply token transforms left to right, skipping None."""
    def transform(tokens):
        for tr in transforms:
            if tr is not None:
                tokens = tr(tokens)
        return tokens
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
