"""Download every dataset and resource used in the project into data/raw/.

Usage:
    python scripts/download_data.py                 # everything
    python scripts/download_data.py imdb blitzer    # only some resources

Raw data is git-ignored and never redistributed. The MPQA lexicon needs a request
form (mpqa.cs.pitt.edu), so it cannot be downloaded automatically: place
subjclueslen1-HLTEMNLP05.tff in data/raw/mpqa/ by hand.
"""
import argparse
import os
import shutil
import sys
import tarfile
import zipfile

import requests
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
BLITZER_URL = "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz"
DRUGS_URL = "https://archive.ics.uci.edu/static/public/462/drug+review+dataset+drugs+com.zip"
FINPHRASE_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/"
                 "resolve/main/data/FinancialPhraseBank-v1.0.zip")


def fetch(url, dest):
    """Stream a file to disk with a progress bar; skip if it already exists."""
    if os.path.exists(dest):
        print(f"  already downloaded: {os.path.relpath(dest, ROOT)}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)
    return dest


def extract(archive, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if archive.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(out_dir, filter="data")
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)


def get_imdb():
    out = os.path.join(RAW, "imdb")
    if os.path.isdir(os.path.join(out, "aclImdb")):
        print("  IMDB already extracted"); return
    extract(fetch(IMDB_URL, os.path.join(RAW, "archives", "aclImdb_v1.tar.gz")), out)


def get_blitzer():
    out = os.path.join(RAW, "blitzer")
    if os.path.isdir(os.path.join(out, "sorted_data_acl")):
        print("  Blitzer already extracted"); return
    extract(fetch(BLITZER_URL, os.path.join(RAW, "archives", "domain_sentiment_data.tar.gz")), out)


def get_drugs():
    out = os.path.join(RAW, "drugs")
    if os.path.exists(os.path.join(out, "drugsComTest_raw.tsv")):
        print("  Drug reviews already extracted"); return
    extract(fetch(DRUGS_URL, os.path.join(RAW, "archives", "drugs.zip")), out)
    # Some versions of the UCI archive contain a nested zip.
    for name in os.listdir(out):
        if name.endswith(".zip"):
            extract(os.path.join(out, name), out)


def get_finphrase():
    out = os.path.join(RAW, "finphrase")
    if os.path.isdir(os.path.join(out, "FinancialPhraseBank-v1.0")):
        print("  Financial PhraseBank already extracted"); return
    # The dataset's loading script no longer works with current `datasets`,
    # so the data zip is downloaded directly.
    extract(fetch(FINPHRASE_URL, os.path.join(RAW, "archives", "FinancialPhraseBank-v1.0.zip")), out)


def get_yelp():
    out = os.path.join(RAW, "yelp", "test.csv")
    if os.path.exists(out):
        print("  Yelp already downloaded"); return
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/yelp_polarity", split="test")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.to_pandas().to_csv(out, index=False)


def get_opinion_lexicon():
    import nltk
    nltk.download("opinion_lexicon", download_dir=os.path.join(RAW, "nltk_data"), quiet=True)


def get_glove():
    # gensim's downloader serves the same GloVe 6B 100d vectors (Wikipedia + Gigaword)
    # as nlp.stanford.edu, in a smaller pre-converted format.
    os.environ.setdefault("GENSIM_DATA_DIR", os.path.join(RAW, "gensim-data"))
    import gensim.downloader as api
    api.load("glove-wiki-gigaword-100", return_path=True)


def check_mpqa():
    path = os.path.join(RAW, "mpqa", "subjclueslen1-HLTEMNLP05.tff")
    if os.path.exists(path):
        print("  MPQA lexicon found")
    else:
        print("  MPQA lexicon missing: request it at mpqa.cs.pitt.edu and save it as "
              f"{os.path.relpath(path, ROOT)}")


RESOURCES = {
    "imdb": get_imdb,
    "blitzer": get_blitzer,
    "yelp": get_yelp,
    "drugs": get_drugs,
    "finphrase": get_finphrase,
    "opinion_lexicon": get_opinion_lexicon,
    "glove": get_glove,
    "mpqa": check_mpqa,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", choices=[*RESOURCES, []], help="resources to fetch (default: all)")
    args = ap.parse_args()
    failed = []
    for name in args.only or RESOURCES:
        print(f"[{name}]")
        try:
            RESOURCES[name]()
        except Exception as e:  # keep going so one slow mirror does not block the rest
            print(f"  FAILED: {e}")
            failed.append(name)
    if failed:
        print("Failed:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
