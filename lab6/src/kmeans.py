import numpy as np


class KMeans:
    def __init__(self, K=3):
        self.K = K
        self.centers = None

    def fit(self, X):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), size=self.K, replace=False)    
        self.centers = X[indices].astype(float)                 

        for _ in range(100):                
            labels = self.predict(X)
            new_centers = self.centers.copy()   

            for k in range(self.K):
                points = X[labels == k]     
                if len(points) > 0:
                    new_centers[k] = points.mean(axis=0)

            converged = np.max(np.abs(self.centers - new_centers)) < 1e-6
            self.centers = new_centers
            if converged:
                break

        return self
    
    def predict(self, X):
        if self.centers is None:
            raise ValueError("Call fit() before predict().")

        labels = []
        for point in X:
            distances = [np.sum((point - center) ** 2) for center in self.centers]      
            labels.append(np.argmin(distances))

        return np.array(labels, dtype=int)
