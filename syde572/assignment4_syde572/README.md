# SYDE 572 Assignment 4 — Code

Naive Bayes sentiment classification of customer reviews.
Report: https://mzhu296.github.io/syde/syde572/assignment4_syde572/ieee.html

## Setup

```bash
conda create -n syde572 --override-channels -c conda-forge python=3.12
conda activate syde572
pip install -r requirements.txt
```

## Run (from this folder)

```bash
python nb.py          # required part: training, sanity checks, accuracy, error analysis
python optional.py    # optional tasks A–G; writes figures to output/
```

| File | Contents |
|---|---|
| `nb.py` | From-scratch Naive Bayes (`train_nb`, `score_doc_label`, `classify_nb`, ...); needs only the standard library |
| `optional.py` | Topic classifier, learning curve, informative words, iris Gaussian NB, scikit-learn, bigram/negation features, domain transfer |
| `SYDE572 assn4 file/all_sentiment_shuffled.txt` | Review data (from the provided zip) |
| `ieee.html` | The assignment report |
