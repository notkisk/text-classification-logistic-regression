# Text Classification with Logistic Regression

A from-scratch **binary sentiment classifier** for movie reviews, built to show what actually
happens between raw text and a prediction. No `sklearn.linear_model` magic — the sigmoid,
the cross-entropy loss, and the gradient-descent updates are all implemented with NumPy, then
benchmarked against a Scikit-Learn baseline.

> **Read the notebook:** [`TextClassificationLogisticRegression.ipynb`](TextClassificationLogisticRegression.ipynb)

---

## Overview

The notebook walks through the full pipeline in order, with unit tests at every step so you
can check your own implementation as you go:

1. **Load** the IMDb movie-review corpus from NLTK and shuffle it.
2. **Vectorize** with a Bag-of-Words model (`CountVectorizer`, 5,000 features, English stop words removed).
3. **Split** 80/20 into train and test sets.
4. **Implement**
   - `sigmoid(z)` — with clipping to avoid `exp` overflow
   - `cross_entropy_loss(y_hat, y)` — with an epsilon to avoid `log(0)`
   - `fit()` — stochastic gradient descent, one sample at a time
   - `predict_proba()` / `predict()`
5. **Evaluate** with accuracy, precision, recall, F1 and a confusion matrix.
6. **Compare** against `sklearn.linear_model.SGDClassifier(loss="log_loss")`.
7. **Deploy** a small `deploy_sentiment_model(text)` helper that scores any raw string.

## The Math

Sigmoid squashes a linear score into a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = w^\top x + b$$

Binary cross-entropy measures how wrong that probability is:

$$L_{CE}(\hat{y}, y) = -\left[ y \log \hat{y} + (1 - y)\log(1-\hat{y}) \right]$$

Deriving the gradient gives a pleasantly simple update rule:

$$w \leftarrow w - \eta\,(\hat{y} - y)\,x, \qquad b \leftarrow b - \eta\,(\hat{y} - y)$$

## Dataset

| | |
|---|---|
| Source | `nltk.corpus.movie_reviews` (IMDb reviews, pre-tokenized) |
| Size | 2,000 documents |
| Classes | 1,000 positive / 1,000 negative (balanced) |
| Split | 1,600 train / 400 test |

The corpus is downloaded automatically the first time the notebook runs:

```python
import nltk
nltk.download("movie_reviews")
```

## Results

Both models are trained on the same feature matrix and evaluated on the same held-out split.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Custom logistic regression (NumPy SGD) | run the notebook | | | |
| Scikit-Learn `SGDClassifier` | run the notebook | | | |

The comparison is deliberately honest: a hand-written SGD loop is **not** expected to beat a
tuned, optimized implementation. What matters is that the numbers land in the same ballpark and
that every moving part is visible.

## Challenges

The notebook closes with three extensions that go beyond the base implementation:

- **Decision threshold** — lowering the cut-off from `0.5` to `0.3` trades precision for recall. Why that trade-off matters for spam filtering.
- **L2 regularization** — a `L2LogisticRegression` subclass that adds `λ·w` to the weight gradient to curb large weights.
- **Mini-batch gradient descent** — a `MiniBatchLogisticRegression` subclass that shuffles each epoch and updates on batches of 32.

## Getting Started

```bash
git clone https://github.com/notkisk/text-classification-logistic-regression.git
cd text-classification-logistic-regression
pip install -r requirements.txt
jupyter notebook TextClassificationLogisticRegression.ipynb
```

Or with a virtual environment, which is recommended:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training the custom model loops over ~16,000 samples in pure Python, so expect the cell to take
a minute or two to finish. Everything runs on CPU; there is no GPU code here.

## Project Structure

```
.
├── README.md
├── requirements.txt
└── TextClassificationLogisticRegression.ipynb
```

## Tech Stack

Python · NumPy · Pandas · NLTK · Scikit-Learn · Matplotlib · Jupyter

## Author

**Noureddine Ben Attous**
MSc Student in Artificial Intelligence and Data Science

Based on the original implementation at
[enori-ben/text-classification-logistic-regression-](https://github.com/enori-ben/text-classification-logistic-regression-).
