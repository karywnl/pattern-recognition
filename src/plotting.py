import matplotlib.pyplot as plt
import numpy as np


def _finish_figure(fig, save_path=None):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def histogram(data, title="", bins=30, xlabel="value", ylabel="frequency"):
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finish_figure(fig)


def show_pipeline(images, titles, save_path=None):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))

    for ax, image, title in zip(axes, images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)

        ax.set_title(title)
        ax.axis("off")

    _finish_figure(fig, save_path)

# show original, reconstructed and error image side by side
def show_reconstruction(original, reconstructed, title="", save_path=None):
    error = np.abs(original - reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("reconstructed")
    axes[2].imshow(error, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(r"$|\mathrm{error}|$")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    _finish_figure(fig, save_path)

# plot E(k) vs k, series = {label: (ks, errors)}
def error_curve(series, title="", save_path=None):
    fig, ax = plt.subplots()
    for label, (ks, errors) in series.items():
        ax.plot(ks, errors, label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(title)
    ax.legend()
    _finish_figure(fig, save_path)
