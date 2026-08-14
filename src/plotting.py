import matplotlib.pyplot as plt
import numpy as np

def histogram(data, title="", bins=30, xlabel="value", ylabel="frequency"):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# show original, reconstructed and error image side by side
def show_reconstruction(original, reconstructed, title=""):
    error = np.abs(original - reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("reconstructed")
    axes[2].imshow(error, cmap="gray")
    axes[2].set_title("error")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.show()

# plot E(k) vs k, series = {label: (ks, errors)}
def error_curve(series, title=""):
    plt.figure()
    for label, (ks, errors) in series.items():
        plt.plot(ks, errors, label=label)
    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.title(title)
    plt.legend()
    plt.show()
