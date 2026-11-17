"""Text preprocessing shared by every domain.

Steps: strip HTML tags, unescape entities, lowercase, tokenize.
Stopwords are deliberately KEPT: standard stopword lists remove "not", "never",
"but" and "n't", which carry exactly the negation and contrast effects studied here.
"""
import html
import re
import string

# Only real tags (<br />, <i>, </i>, <hr>, ...): a letter or "/" must follow "<", so
# emoticons such as ">.<" and "</3" are kept as text, since they carry sentiment.
_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
# Tokens, in order of preference: the clitic "n't"; numbers with decimals ("9.99");
# words (any Unicode letters, so "café" stays whole) with internal apostrophes/hyphens;
# or a single punctuation mark (kept, because punctuation bounds negation scope).
_TOKEN = re.compile(r"n't|\d+(?:[.,]\d+)*|[^\W_]+(?:[-'][^\W_]+)*|[^\w\s]|_")
_NT_SPLIT = re.compile(r"([^\W\d_]+)n't\b")
_IRREGULAR_NT = {"can't": "ca n't", "won't": "wo n't", "shan't": "sha n't", "ain't": "ai n't"}
# Windows-1252 characters that survive as C1 control codes in some IMDB reviews,
# plus typographic quotes and dashes, normalised to ASCII. Dashes get spaces so that
# "good<dash>really" is not glued into one hyphenated word.
_CHARMAP = str.maketrans({"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
                          "\x96": " - ", "\x97": " - ", "\x85": "...",
                          "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                          "\u2013": " - ", "\u2014": " - ", "\u2026": "..."})


def clean(text):
    """Remove HTML tags, unescape entities like &amp; and normalise quotes/dashes."""
    text = html.unescape(_TAG.sub(" ", text))
    return text.translate(_CHARMAP)


def tokenize(text):
    """Lowercase and tokenize, splitting the clitic n't off its verb (Penn Treebank style).

    >>> tokenize("I didn't like it.<br /><br />Not GOOD!")
    ['i', 'did', "n't", 'like', 'it', '.', 'not', 'good', '!']
    """
    text = clean(text).lower()
    for k, v in _IRREGULAR_NT.items():
        text = text.replace(k, v)
    text = _NT_SPLIT.sub(r"\1 n't", text)
    return _TOKEN.findall(text)


def is_punct(tok):
    return all(ch in string.punctuation for ch in tok)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
