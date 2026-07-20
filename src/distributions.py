import numpy as np

# Uniform Distribution
def uniform(a, b, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    samples = np.zeros(n)
    for i in range(n):
        u = rng.random()          # raw sample, uniform in [0, 1)
        samples[i] = a + (b - a) * u   # rescale u into [a, b)

    return samples


