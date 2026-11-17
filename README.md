# Cross-Domain Sentiment Analysis with Naive Bayes, Lexicons and Word Embeddings

CL2 course project (P21). A multinomial Naive Bayes classifier is trained on IMDB movie
reviews and tested on seven other domains, to measure which feature designs help and
how well they transfer.

## Research questions

1. Which feature designs improve Naive Bayes over a plain bag-of-words baseline:
   polarity features from two sentiment lexicons (Hu & Liu, MPQA), a lexicon expanded
   with GloVe embeddings, or negation handling? Which negation method works best?
2. How well does each feature design transfer from movie reviews to other domains, and
   does the accuracy drop grow with the distance between domains?
3. How do word polarities compare across the lexicons, the embeddings and each domain's
   own data, and which linguistic phenomena explain the remaining errors?

## Repository layout

```
sentiment/          library code (preprocessing, data loading, classifier, evaluation)
scripts/            command-line entry points (download data, run experiments)
results/            experiment outputs (JSON / CSV tables, figures)
data/raw/           downloaded datasets (git-ignored, created by the download script)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Usage

```bash
python scripts/download_data.py      # fetch all datasets into data/raw/
python scripts/dataset_stats.py      # sizes, class balance, text length -> results/dataset_stats.csv
python scripts/run_baseline.py       # IMDB baseline -> results/baseline_imdb.json
python tests/test_core.py            # unit tests
```

## Data and licenses

Raw data is never committed to this repository; the download script fetches every
resource from its official source. All resources are free for non-commercial academic
research.

| Resource | License / terms | Source |
|---|---|---|
| IMDB Large Movie Review Dataset (Maas et al., 2011) | Free for research with citation | ai.stanford.edu/~amaas/data/sentiment (mirror: HF `stanfordnlp/imdb`) |
| Multi-Domain Sentiment Dataset (Blitzer et al., 2007) | Free for research; cite ACL 2007 | cs.jhu.edu/~mdredze/datasets/sentiment |
| Yelp Review Polarity (Zhang et al., 2015) | Yelp dataset terms, non-commercial academic use | HF `fancyzhx/yelp_polarity` |
| Drug Review Dataset, Drugs.com (Gräßer et al., 2018) | CC BY 4.0; research only, no redistribution | UCI ML Repository, dataset 462 |
| Financial PhraseBank (Malo et al., 2014) | CC BY-NC-SA 3.0 | HF `takala/financial_phrasebank` (data zip) |
| Opinion Lexicon (Hu & Liu, 2004) | Free; distributed with NLTK | NLTK `opinion_lexicon` |
| MPQA Subjectivity Lexicon (Wilson et al., 2005) | GNU GPL; request form | mpqa.cs.pitt.edu (place file manually, see below) |
| GloVe 6B, 100d (Pennington et al., 2014) | PDDL 1.0 | nlp.stanford.edu/projects/glove |

**Availability notes (checked 17 Nov 2026).** ai.stanford.edu timed out from our network,
so the download script falls back to the Hugging Face mirror `stanfordnlp/imdb` (same
25k/25k labeled reviews). UCI has withdrawn the Drugs.com dataset (id 462) from public
download; because its terms forbid redistribution, unofficial mirrors are not used, and
the drug domain is pending until the files are obtained from the authors. The MPQA lexicon
must be placed in `data/raw/mpqa/` by hand after the request form.

## Progress

- [x] Week 1 – download scripts, preprocessing, validation split, IMDB baseline

## Results so far

### Datasets after preprocessing (`results/dataset_stats.csv`)

| Domain | Raw pos / neg | Used (balanced) | Mean tokens |
|---|---|---|---|
| IMDB train | 12,500 / 12,500 | 20,000 | 267 |
| IMDB validation | (held out from train) | 5,000 | 270 |
| IMDB test | 12,500 / 12,500 | 25,000 | 262 |
| Amazon books | 1,000 / 1,000 | 2,000 | 198 |
| Amazon DVD | 1,000 / 1,000 | 2,000 | 193 |
| Amazon electronics | 1,000 / 1,000 | 2,000 | 125 |
| Amazon kitchen | 1,000 / 1,000 | 2,000 | 105 |
| Yelp | 19,000 / 19,000 | 2,000 | 163 |
| Financial PhraseBank (50% agree) | 1,363 / 604 | 1,208 | 27 |

### Baseline: multinomial NB, unigram counts, add-one smoothing (`results/baseline_imdb.json`)

| IMDB split | Accuracy | Macro F1 | P / R (pos) | P / R (neg) |
|---|---|---|---|---|
| Validation (5,000) | 0.834 | 0.834 | 0.862 / 0.795 | 0.810 / 0.873 |
| Test (25,000) | 0.809 | 0.808 | 0.855 / 0.745 | 0.774 / 0.873 |

The baseline over-predicts the negative class (recall 0.87 vs 0.75). Its most polar words
(`results/baseline_top_features.csv`) are mostly names (e.g. *seagal*, *boll* negative;
*paulie*, *edie* positive): movie-specific vocabulary that cannot transfer to other
domains, which motivates the lexicon features of Week 2.
- [ ] Week 2 – Hu & Liu and MPQA lexicon features, GloVe-expanded lexicon
- [ ] Week 3 – negation methods, IMDB ablation, cross-domain tests
- [ ] Week 4 – domain distances, word polarities, confidence intervals, significance tests
- [ ] Week 5 – error analysis, agreement check, report

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. *Proceedings of ACL 2007*, 440–447.
- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- Councill, I. G., McDonald, R., & Velikovich, L. (2010). What's Great and What's Not: Learning to Classify the Scope of Negation for Improved Sentiment Analysis. *Proceedings of the Workshop on Negation and Speculation in NLP*, 51–59.
- Das, S. R., & Chen, M. Y. (2001). Yahoo! for Amazon: Extracting Market Sentiment from Stock Message Boards. *Proceedings of APFA 2001*.
- Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. *Proceedings of the 2018 International Conference on Digital Health*, 121–125.
- Hamilton, W. L., Clark, K., Leskovec, J., & Jurafsky, D. (2016). Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. *Proceedings of EMNLP 2016*, 595–605.
- Harris, C. R., et al. (2020). Array Programming with NumPy. *Nature*, 585, 357–362.
- Hu, M., & Liu, B. (2004). Mining and Summarizing Customer Reviews. *Proceedings of ACM SIGKDD 2004*, 168–177.
- Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90–95.
- Kallumadi, S., & Gräßer, F. (2018). Drug Reviews (Drugs.com) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SK5S
- Lhoest, Q., et al. (2021). Datasets: A Community Library for Natural Language Processing. *Proceedings of EMNLP 2021: System Demonstrations*.
- Lin, J. (1991). Divergence Measures Based on the Shannon Entropy. *IEEE Transactions on Information Theory*, 37(1), 145–151.
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. *Proceedings of ACL 2011*, 142–150.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. *JASIST*, 65(4), 782–796.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.
- McNemar, Q. (1947). Note on the Sampling Error of the Difference between Correlated Proportions or Percentages. *Psychometrika*, 12(2), 153–157.
- Pan, S. J., Ni, X., Sun, J.-T., Yang, Q., & Chen, Z. (2010). Cross-Domain Sentiment Classification via Spectral Feature Alignment. *Proceedings of WWW 2010*, 751–760.
- Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis. *Foundations and Trends in Information Retrieval*, 2(1–2), 1–135.
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment Classification using Machine Learning Techniques. *Proceedings of EMNLP 2002*, 79–86.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of EMNLP 2014*, 1532–1543.
- Řehůřek, R., & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45–50.
- Taboada, M., Brooke, J., Tofiloski, M., Voll, K., & Stede, M. (2011). Lexicon-Based Methods for Sentiment Analysis. *Computational Linguistics*, 37(2), 267–307.
- Turney, P. D. (2002). Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews. *Proceedings of ACL 2002*, 417–424.
- Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261–272.
- Wang, S., & Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. *Proceedings of ACL 2012*, 90–94.
- Wang, S., Lv, G., Mazumder, S., & Liu, B. (2021). Detecting Domain Polarity-Changes of Words in a Sentiment Lexicon. *Findings of ACL-IJCNLP 2021*, 3657–3668.
- Wiegand, M., Balahur, A., Roth, B., Klakow, D., & Montoyo, A. (2010). A Survey on the Role of Negation in Sentiment Analysis. *Proceedings of the Workshop on Negation and Speculation in NLP*, 60–68.
- Wilson, T., Wiebe, J., & Hoffmann, P. (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. *Proceedings of HLT/EMNLP 2005*, 347–354.
- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *NeurIPS 2015*, 649–657.
