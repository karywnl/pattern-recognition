import matplotlib.pyplot as plt

def histogram(data, title="", bins=30, xlabel="value", ylabel="frequency"):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
