import matplotlib.pyplot as plt


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
