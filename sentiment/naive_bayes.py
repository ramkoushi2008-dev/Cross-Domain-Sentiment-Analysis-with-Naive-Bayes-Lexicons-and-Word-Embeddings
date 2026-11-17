"""Multinomial Naive Bayes with add-alpha smoothing (Jurafsky & Martin, Fig. 4.2).

Carried over from my CL2 Assignment 3 implementation. Features are Counters of tokens;
new feature designs (lexicon tokens, NOT_ prefixes) are added by appending marker tokens
to each document, so this classifier never changes between experiments.
"""
import math
from collections import Counter


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """X: list of Counter(token -> count); y: list of labels."""
        self.classes = sorted(set(y))
        n_docs = len(y)
        self.logprior = {c: math.log(y.count(c) / n_docs) for c in self.classes}

        self.counts = {c: Counter() for c in self.classes}
        for feats, label in zip(X, y):
            self.counts[label].update(feats)

        self.V = set()
        for c in self.classes:
            self.V |= set(self.counts[c])
        self.total = {c: sum(self.counts[c].values()) for c in self.classes}

        self.loglik = {c: {} for c in self.classes}
        for c in self.classes:
            denom = self.total[c] + self.alpha * len(self.V)
            for w in self.V:
                self.loglik[c][w] = math.log((self.counts[c][w] + self.alpha) / denom)
        return self

    def scores(self, feats):
        """Unnormalised log posterior for each class; unknown words are skipped."""
        out = {}
        for c in self.classes:
            s = self.logprior[c]
            for w, n in feats.items():
                if w in self.V:
                    s += n * self.loglik[c][w]
            out[c] = s
        return out

    def predict_one(self, feats):
        sc = self.scores(feats)
        return max(sc, key=sc.get)

    def predict(self, X):
        return [self.predict_one(f) for f in X]

    def log_ratio(self, word, pos=1, neg=0):
        """log P(w|pos) - log P(w|neg): the learned polarity of a word (used in RQ3)."""
        return self.loglik[pos][word] - self.loglik[neg][word]

    def top_features(self, k=20, pos=1, neg=0, min_count=0):
        """The k most positive and k most negative words by log ratio."""
        words = [w for w in self.V
                 if self.counts[pos][w] + self.counts[neg][w] >= min_count]
        ranked = sorted(words, key=lambda w: self.log_ratio(w, pos, neg))
        return ranked[::-1][:k], ranked[:k]
