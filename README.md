# Linear Models for Classification: Research Assignment

Research-level assignment from the **Machine Learning** course at Lomonosov Moscow State University (CMC faculty, 2025/2026): a study of logistic regression and categorical feature encoding.

## What's inside

**`Task.py`**: my own implementations, covered by unit tests:
- `MyOneHotEncoder`: one-hot encoding from scratch
- `SimpleCounterEncoder`: *counter* (target/mean) encoding of categorical features with smoothing
- `FoldCounters`: out-of-fold counter encoding that prevents target leakage
- `weights`: closed-form optimal weights for a one-hot-encoded categorical feature

**`linear_models_classification.ipynb`** covers:
- convergence of first-order (L-BFGS) vs Newton-type optimizers on unscaled data;
- how feature scaling affects iteration count and model quality;
- tuning the regularization strength on a validation set, and why not on the train or test set;
- one-hot vs counter encoding of categorical features, compared by ROC-AUC, and the trade-offs between them;
- comparison with CatBoost.

## Stack
Python, NumPy, pandas, scikit-learn, CatBoost, Matplotlib, Seaborn

> The notebook is written in Russian.
