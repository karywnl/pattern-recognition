import numpy as np


def dtw(x, y, window=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    n, m = len(x), len(y)
    if n == 0 or m == 0 or x.shape[1] != y.shape[1]:
        raise ValueError("Use nonempty sequences with matching feature counts.")
    width = max(n, m) if window is None else max(window, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0

    for i in range(n):
        distances = np.sqrt(np.sum((y - x[i]) ** 2, axis=1))
        for j in range(max(0, i - width), min(m, i + width + 1)):
            cost[i + 1, j + 1] = distances[j] + min(
                cost[i, j], cost[i, j + 1], cost[i + 1, j])
    return cost[n, m]


class DTW:
    def __init__(self, references_per_class=10, window=25):
        self.references_per_class = references_per_class
        self.window = window

    def fit(self, sequences, y):
        y = np.asarray(y)
        rng = np.random.default_rng(42)
        self.indices = []
        for label in np.unique(y):
            candidates = np.flatnonzero(y == label)
            count = min(self.references_per_class, len(candidates))
            self.indices.extend(rng.choice(candidates, count, replace=False))
        self.sequences = [sequences[i] for i in self.indices]
        self.y = y[self.indices]
        return self

    def predict(self, sequences):
        predictions = []
        for sequence in sequences:
            distances = [dtw(sequence, reference, self.window)
                         for reference in self.sequences]
            predictions.append(self.y[np.argmin(distances)])
        return np.array(predictions)
