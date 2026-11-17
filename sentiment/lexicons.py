"""Polarity lexicons: Hu & Liu (2004) Opinion Lexicon and MPQA Subjectivity Lexicon (Wilson et al., 2005).

Each loader returns a dict word -> +1 (positive) or -1 (negative).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
HU_LIU_DIR = os.path.join(RAW, "nltk_data", "corpora", "opinion_lexicon")
MPQA_PATH = os.path.join(RAW, "mpqa", "subjclueslen1-HLTEMNLP05.tff")


def _read_word_list(path):
    words = []
    with open(path, encoding="utf-8") as f:  # ASCII except one UTF-8 entry ("naïve")
        for line in f:
            line = line.strip()
            if line and not line.startswith(";"):
                words.append(line.lower())
    return words


def load_hu_liu(directory=HU_LIU_DIR):
    """Hu & Liu Opinion Lexicon: about 2,000 positive and 4,800 negative words.

    The few words listed under both polarities (e.g. "envious") are dropped as ambiguous.
    """
    pos = set(_read_word_list(os.path.join(directory, "positive-words.txt")))
    neg = set(_read_word_list(os.path.join(directory, "negative-words.txt")))
    both = pos & neg
    lex = {w: 1 for w in pos - both}
    lex.update({w: -1 for w in neg - both})
    return lex


def parse_mpqa_line(line):
    """'type=strongsubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative'
    -> {'type': 'strongsubj', 'word1': 'abandoned', ...}"""
    fields = {}
    for part in line.split():
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k] = v
    return fields


def load_mpqa(path=MPQA_PATH):
    """MPQA Subjectivity Lexicon, positive and negative prior polarities only.

    Entries are per (word, part of speech); neutral and "both" entries are skipped, and a
    word whose entries disagree in polarity across parts of speech is dropped as ambiguous.
    Words are matched literally (the stemmed=y flag is not expanded), the same as Hu & Liu.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MPQA lexicon not found at {path}. Request it at mpqa.cs.pitt.edu and save "
            "subjclueslen1-HLTEMNLP05.tff there.")
    polarities = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            e = parse_mpqa_line(line)
            pol = {"positive": 1, "negative": -1}.get(e.get("priorpolarity"))
            if pol is not None and "word1" in e:
                polarities.setdefault(e["word1"].lower(), set()).add(pol)
    return {w: next(iter(p)) for w, p in polarities.items() if len(p) == 1}


def coverage(lexicon, docs_tokens):
    """Share of tokens and of documents that contain at least one lexicon word."""
    n_tok = hit_tok = hit_doc = 0
    for toks in docs_tokens:
        hits = sum(1 for t in toks if t in lexicon)
        n_tok += len(toks)
        hit_tok += hits
        hit_doc += hits > 0
    return {"token_coverage": hit_tok / n_tok, "doc_coverage": hit_doc / len(docs_tokens)}
