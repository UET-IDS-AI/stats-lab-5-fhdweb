import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    np.random.seed(42)
    samples = np.random.exponential(scale=1/1, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x-mu)**2/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    PA = 0.3
    PB = 0.7

    likelihood_A = np.exp(-(time-40)**2 / 4)
    likelihood_B = np.exp(-(time-45)**2 / 4)

    numerator = PB * likelihood_B
    denominator = PA * likelihood_A + PB * likelihood_B

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    np.random.seed(42)

    groups = np.random.choice(["A", "B"], size=n, p=[0.3, 0.7])

    times = np.zeros(n)

    times[groups == "A"] = np.random.normal(40, 2, np.sum(groups == "A"))
    times[groups == "B"] = np.random.normal(45, 2, np.sum(groups == "B"))

    mask = np.abs(times - time) < 0.5

    if np.sum(mask) == 0:
        return 0

    return np.mean(groups[mask] == "B")
