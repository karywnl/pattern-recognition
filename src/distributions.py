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


# Gaussian Distribution (polar / Marsaglia method)
def gaussian(mu, sigma, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    samples = np.zeros(n)
    i = 0
    while i < n:
        u1 = uniform(-1, 1, 1, rng)[0]
        u2 = uniform(-1, 1, 1, rng)[0]
        s = u1 ** 2 + u2 ** 2

        if not (0 < s < 1):
            continue             # rejected -- go throw another dart

        k = np.sqrt(-2 * np.log(s) / s)
        x = u1 * k
        y = u2 * k

        samples[i] = mu + sigma * x
        i += 1
        if i < n:
            samples[i] = mu + sigma * y
            i += 1

    return samples
