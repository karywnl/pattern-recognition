import time
import numpy as np


# Linear congruential generator: keeps its state between calls.
class LCG:
    def __init__(self, seed=None):
        if seed is None:
            seed = time.time_ns()
        self.state = int(seed) % (2 ** 64)


# Uniform Distribution
def uniform(a, b, n, rng=None):
    if rng is None:
        rng = LCG()

    multiplier = 6364136223846793005
    increment = 1442695040888963407
    modulus = 2 ** 64

    samples = np.zeros(n)
    for i in range(n):
        rng.state = (multiplier * rng.state + increment) % modulus
        # Convert the state to [0, 1), using 53 bits to avoid rounding to 1.
        u = (rng.state // (2 ** 11)) / (2 ** 53)
        samples[i] = a + (b - a) * u

    return samples


# Gaussian Distribution (polar / Marsaglia method)
def gaussian(mu, sigma, n, rng=None):
    if rng is None:
        rng = LCG()

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
