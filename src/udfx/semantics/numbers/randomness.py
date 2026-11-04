import random

GAUSS = random.gauss


def gauss(mu: float = 0.0, sigma: float = 1.0) -> float:
    return GAUSS(mu, sigma)
