import numpy as np


class MedianImputerKeepAll:
    """Median imputer that preserves feature count even if a column is entirely NaN.

    This is intentionally implemented as a real module-level class so it can be pickled
    and later loaded in production inference.
    """

    def __init__(self, fill_value_if_all_nan: float = 0.0):
        self.fill_value_if_all_nan = float(fill_value_if_all_nan)
        self.statistics_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        stats = []
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            finite = col[~np.isnan(col)]
            if finite.size == 0:
                stats.append(self.fill_value_if_all_nan)
            else:
                stats.append(float(np.median(finite)))
        self.statistics_ = np.asarray(stats, dtype=float)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise ValueError("Imputer not fitted")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.shape[1] != self.statistics_.shape[0]:
            raise ValueError(
                f"Feature count mismatch: X has {X_arr.shape[1]} features, imputer expects {self.statistics_.shape[0]}"
            )
        mask = np.isnan(X_arr)
        if mask.any():
            X_arr = X_arr.copy()
            X_arr[mask] = np.take(self.statistics_, np.where(mask)[1])
        return X_arr

    def fit_transform(self, X):
        return self.fit(X).transform(X)
