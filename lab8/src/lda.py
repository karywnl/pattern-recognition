import numpy as np


class LDA:
    def __init__(self, reg=1e-3):
        self.reg = reg

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        means = []
        priors = []
        covariance = np.zeros((d, d))

        for label in self.classes:
            samples = X[y == label]
            mean = samples.mean(axis=0)
            means.append(mean)
            priors.append(len(samples) / n)
            difference = samples - mean
            covariance += difference.T @ difference

        self.means = np.array(means)
        self.covariance = covariance / n + self.reg * np.eye(d)
        self.weights = np.linalg.solve(self.covariance, self.means.T).T
        self.bias = -0.5 * np.sum(self.means * self.weights, axis=1) + np.log(priors)
        return self

    def predict(self, X):
        scores = X @ self.weights.T + self.bias
        return self.classes[np.argmax(scores, axis=1)]
