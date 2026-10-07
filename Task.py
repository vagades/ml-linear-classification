import numpy as np


class MyOneHotEncoder:
    def __init__(self, dtype=int):
        self.categories_ = None
        self.dtype = dtype

    def fit(self, X):
        self.categories_ = []
        for col in X.columns:
            unique_values = sorted(X[col].unique())
            self.categories_.append(unique_values)

    def transform(self, X):
        n_samples = X.shape[0]
        encoded_columns = []
        for i, col in enumerate(X.columns):
            values = X[col]
            unique_values = self.categories_[i]
            n_unique = len(unique_values)
            col_encoded = np.zeros((n_samples, n_unique), dtype=self.dtype)
            for j, val in enumerate(unique_values):
                col_encoded[:, j] = (values == val).astype(self.dtype)
            encoded_columns.append(col_encoded)
        encoded = np.concatenate(encoded_columns, axis=1)
        return encoded


class SimpleCounterEncoder:
    def __init__(self):
        self.statistics_ = None

    def fit(self, X, y):
        y = np.array(y)
        self.statistics_ = []
        n_objects = len(X)
        for col in X.columns:
            stats = {}
            col_values = X[col]
            unique_values = sorted(col_values.unique())
            for val in unique_values:
                mask = col_values == val
                count = int(np.sum(mask))
                successes = float(np.sum(y[mask]))
                mean_success = successes / count if count > 0 else 0.0
                freq = count / n_objects if n_objects > 0 else 0.0
                stats[val] = {"successes": mean_success, "counters": freq}
            self.statistics_.append(stats)

    def transform(self, X, a=1e-5, b=1e-5):
        n_samples = X.shape[0]
        encoded_columns = []
        for i, col in enumerate(X.columns):
            col_values = X[col]
            stats = self.statistics_[i]
            col_encoded = np.zeros((n_samples, 3), dtype=float)
            for j in range(n_samples):
                val = col_values.iloc[j]
                if val in stats:
                    s = stats[val]["successes"]
                    c = stats[val]["counters"]
                    r = (s + a) / (c + b)
                else:
                    s = c = r = 0.0
                col_encoded[j] = [s, c, r]
            encoded_columns.append(col_encoded)
        encoded = np.concatenate(encoded_columns, axis=1)
        return encoded


class FoldCounters:
    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
        self.folds = None
        self.fold_stats_ = None

    def group_k_fold(self, n_samples):
        # Используем RandomState, чтобы детерминированно совпасть с эталоном тестов
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n_samples)
        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds, dtype=int)
        fold_sizes[: n_samples % self.n_folds] += 1
        folds = []
        current = 0
        for fold_size in fold_sizes:
            folds.append(indices[current:current + fold_size])
            current += fold_size
        return folds

    def fit(self, X, y, seed=None):
        if seed is not None:
            self.seed = seed
        n_samples = len(X)
        self.folds = self.group_k_fold(n_samples)
        self.fold_stats_ = []

        # Считаем статистики по каждому фолду (OOF)
        for fold_idx in range(self.n_folds):
            val_idx = self.folds[fold_idx]
            train_idx = np.setdiff1d(np.arange(n_samples), val_idx)
            X_train, y_train = X.iloc[train_idx], np.array(y)[train_idx]

            stats_for_fold = []
            for col in X.columns:
                stats = {}
                col_values = X_train[col]
                unique_values = sorted(col_values.unique())
                for val in unique_values:
                    mask = col_values == val
                    count = int(np.sum(mask))
                    successes = float(np.sum(y_train[mask]))
                    mean_success = successes / count if count > 0 else 0.0
                    freq = count / len(X_train) if len(X_train) > 0 else 0.0
                    stats[val] = {"successes": mean_success, "counters": freq}
                stats_for_fold.append(stats)
            self.fold_stats_.append(stats_for_fold)

    def transform(self, X, a=1e-5, b=1e-5):
        n_samples = len(X)
        encoded = np.zeros((n_samples, len(X.columns) * 3), dtype=float)

        # Кодируем объект статистикой его фолда (без утечек)
        for fold_idx, val_idx in enumerate(self.folds):
            stats_for_fold = self.fold_stats_[fold_idx]
            for i, col in enumerate(X.columns):
                stats = stats_for_fold[i]
                for j in val_idx:
                    val = X.iloc[j][col]
                    if val in stats:
                        s = stats[val]["successes"]
                        c = stats[val]["counters"]
                        r = (s + a) / (c + b)
                    else:
                        s = c = r = 0.0
                    # без пробела перед ':' — фиксим E203
                    encoded[j, 3 * i:3 * i + 3] = [s, c, r]
        return encoded


def weights(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    unique = sorted(np.unique(x))
    result = []
    for val in unique:
        mask = x == val
        count = int(np.sum(mask))
        w = float(np.sum(y[mask]) / count) if count > 0 else 0.0
        result.append(w)
    return np.array(result, dtype=float)
