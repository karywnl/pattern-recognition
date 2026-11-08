import numpy as np


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        predictions = []
        for point in X:
            distances = np.sum((self.X - point) ** 2, axis=1)
            nearest = np.argsort(distances)[:self.k]
            labels, counts = np.unique(self.y[nearest], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)
