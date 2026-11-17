"""Text preprocessing shared by every domain.

Steps: strip HTML tags, unescape entities, lowercase, tokenize.
Stopwords are deliberately KEPT: standard stopword lists remove "not", "never",
"but" and "n't", which carry exactly the negation and contrast effects studied here.
"""
import html
import re
import string

_TAG = re.compile(r"<[^>]+>")
# Tokens: words with internal apostrophes/hyphens, the clitic "n't", numbers,
# or single punctuation marks (punctuation is kept because it bounds negation scope).
_TOKEN = re.compile(r"n't|[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\sa-z0-9]")
_NT_SPLIT = re.compile(r"([a-z]+)n't\b")
_IRREGULAR_NT = {"can't": "ca n't", "won't": "wo n't", "shan't": "sha n't", "ain't": "ai n't"}


def clean(text):
    """Remove HTML tags (IMDB uses <br />) and unescape entities like &amp;."""
    text = _TAG.sub(" ", text)
    return html.unescape(text)


def tokenize(text):
    """Lowercase and tokenize, splitting the clitic n't off its verb (Penn Treebank style).

    >>> tokenize("I didn't like it.<br /><br />Not GOOD!")
    ['i', 'did', "n't", 'like', 'it', '.', 'not', 'good', '!']
    """
    text = clean(text).lower().replace("’", "'")
    for k, v in _IRREGULAR_NT.items():
        text = text.replace(k, v)
    text = _NT_SPLIT.sub(r"\1 n't", text)
    return _TOKEN.findall(text)


def is_punct(tok):
    return all(ch in string.punctuation for ch in tok)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
