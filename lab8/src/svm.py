import numpy as np


class SVM:
    def __init__(self, sigma=None, tol=1e-3, max_iter=100000):
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter

    def kernel(self, X, Z):
        distances = (np.sum(X ** 2, axis=1)[:, None]
                     + np.sum(Z ** 2, axis=1) - 2 * X @ Z.T)
        np.maximum(distances, 0, out=distances)
        distances *= -1 / (2 * self.width ** 2)
        np.exp(distances, out=distances)
        return distances

    def fit_binary(self, X, y, K):
        alpha = np.zeros(len(y))
        gradient = y.astype(float).copy()

        for _ in range(self.max_iter):
            upper = (y == 1) | (alpha > 0)
            lower = (y == -1) | (alpha > 0)
            i = np.argmax(np.where(upper, gradient, -np.inf))
            j = np.argmin(np.where(lower, gradient, np.inf))
            gap = gradient[i] - gradient[j]
            if gap <= self.tol:
                gradient = y - K @ (alpha * y)
                if gradient[upper].max() - gradient[lower].min() <= self.tol:
                    break
                continue

            limit = min(alpha[i] if y[i] == -1 else np.inf,
                        alpha[j] if y[j] == 1 else np.inf)
            curvature = K[i, i] + K[j, j] - 2 * K[i, j]
            if curvature <= 0:
                if np.isinf(limit):
                    raise ValueError("Hard-margin separation failed at this kernel width.")
                step = limit
            else:
                step = min(gap / curvature, limit)
            alpha[i] += y[i] * step
            alpha[j] -= y[j] * step
            gradient -= step * (K[:, i] - K[:, j])

        gradient = y - K @ (alpha * y)
        upper = (y == 1) | (alpha > 0)
        lower = (y == -1) | (alpha > 0)
        gap = gradient[upper].max() - gradient[lower].min()
        support = alpha > 0
        bias = gradient[support].mean()
        return X[support], alpha[support] * y[support], bias, gap

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes = np.unique(y)
        self.width = self.sigma
        if self.width is None:
            self.width = np.sqrt(X.var(axis=0).sum()) / 2
        if self.width <= 0:
            raise ValueError("sigma must be positive.")
        self.models = []
        self.gaps = []

        K = self.kernel(X, X)
        np.fill_diagonal(K, 1.0)
        targets = self.classes if len(self.classes) > 2 else self.classes[:1]
        for label in targets:
            labels = np.where(y == label, 1, -1)
            points, coefficients, bias, gap = self.fit_binary(X, labels, K)
            self.models.append((points, coefficients, bias))
            self.gaps.append(gap)
        self.converged = all(gap <= self.tol for gap in self.gaps)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([self.kernel(X, points) @ coefficients + bias
                                  for points, coefficients, bias in self.models])
        if len(self.classes) == 2:
            return np.where(scores[:, 0] >= 0, self.classes[0], self.classes[1])
        return self.classes[np.argmax(scores, axis=1)]
