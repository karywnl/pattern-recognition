import numpy as np
import matplotlib.pyplot as plt
from .pca import PCA


def plot_pca_2d(X, y):
    pca = PCA(k=2).fit(X)
    points = pca.transform(X)

    plt.figure(figsize=(7, 5))
    for label in np.unique(y):
        selected = y == label
        plt.scatter(points[selected, 0], points[selected, 1],
                    s=10, alpha=0.5, edgecolors="none", label=str(label))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Training data (2D PCA)")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()
