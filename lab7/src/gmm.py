import numpy as np
from .kmeans import KMeans


class GMM:
    def __init__(self, K=3, reg=1e-6, max_iter=100, tol=1e-6):
        self.K = K
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihoods = []

    def component_scores(self, X):
        n, d = X.shape
        scores = np.zeros((n, self.K))

        for k in range(self.K):
            difference = X - self.means[k]
            # Cholesky factor: covariance = cov_sqrt @ cov_sqrt.T
            cov_sqrt = np.linalg.cholesky(self.covariances[k])
            standard_difference = np.linalg.solve(cov_sqrt, difference.T).T
            log_det = 2 * np.log(np.diag(cov_sqrt)).sum()
            d_squared = np.sum(standard_difference ** 2, axis=1)
            scores[:, k] = np.log(self.weights[k]) - 0.5 * (
                d * np.log(2 * np.pi) + log_det + d_squared
            )

        return scores

    def score_samples(self, X):
        # log of the mixture density, calculated without tiny exponentials
        scores = self.component_scores(X)
        largest = scores.max(axis=1)
        return largest + np.log(np.exp(scores - largest[:, None]).sum(axis=1))

    def fit(self, X):
        n, d = X.shape
        model = KMeans(K=self.K)
        model.fit(X)
        labels = model.predict(X)

        self.means = model.centers.copy()
        self.weights = np.zeros(self.K)
        self.covariances = np.zeros((self.K, d, d))
        self.log_likelihoods = []

        for k in range(self.K):
            points = X[labels == k]
            if len(points) == 0:
                raise ValueError("K-means produced an empty component. Try another K.")
            difference = points - self.means[k]
            self.weights[k] = len(points) / n
            self.covariances[k] = difference.T @ difference / len(points)
            self.covariances[k] += self.reg * np.eye(d)

        for _ in range(self.max_iter):
            # E-step: soft membership of each point in each component
            scores = self.component_scores(X)
            scores -= scores.max(axis=1, keepdims=True)
            responsibilities = np.exp(scores)
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: update weights, means, and covariances
            counts = responsibilities.sum(axis=0)
            if np.any(counts < 1e-10):
                raise ValueError("A component has no effective points. Try a smaller K.")
            self.weights = counts / n
            self.means = responsibilities.T @ X / counts[:, None]

            for k in range(self.K):
                difference = X - self.means[k]
                weighted_difference = responsibilities[:, k, None] * difference
                self.covariances[k] = difference.T @ weighted_difference / counts[k]
                self.covariances[k] += self.reg * np.eye(d)

            log_likelihood = self.score_samples(X).mean()
            self.log_likelihoods.append(log_likelihood)
            if len(self.log_likelihoods) > 1:
                change = self.log_likelihoods[-1] - self.log_likelihoods[-2]
                if abs(change) < self.tol:
                    break

        return self
