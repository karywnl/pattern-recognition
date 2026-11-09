import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        if not 1 <= self.k <= X.shape[1]:
            raise ValueError("k must be between 1 and the number of features.")
        self.mean = X.mean(axis=0)
        centered = X - self.mean
        covariance = centered.T @ centered / len(X)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, order[:self.k]]
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components
