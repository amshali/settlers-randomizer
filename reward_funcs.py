import numpy as np


def reward1(variance):
    if variance < 2:
        return 10
    else:
        return -variance


def reward1_d(variance):
    if variance < 2:
        return 1
    else:
        return -variance


def reward1_a(variance):
    if variance < 2:
        return 10
    else:
        return -variance * 10


def reward1_b(variance):
    if variance < 1.75:
        return 10
    else:
        return -variance


def reward1_c(variance):
    if variance < 1.75:
        return 10
    else:
        return -variance * 10


def reward2(variance):
    if variance < 3:
        return 10
    else:
        return -variance


def reward3(variance):
    if variance < 2:
        return 10
    else:
        return -10


def reward4(variance):
    if variance < 3:
        return 10
    else:
        return -10


def reward5(variance):
    return -variance


def reward6(variance):
    if variance < 2:
        return 1 / variance
    else:
        return -variance


def reward7(variance):
    if variance < 2:
        return 10 / variance
    else:
        return -variance


def reward8(variance):
    if variance < 3:
        return 100
    else:
        return -variance


def reward9(variance):
    return 1 / variance


def reward10(variance):
    return 1 / (1 + variance)  # Keeps rewards positive and bounded


def reward11(variance):
    if variance < 2:
        return 1
    else:
        return -1


def reward12(variance):
    if variance < 2:
        return 2
    else:
        return -2


def reward_exponential(variance):
    return np.exp(-variance)


def reward_mean_var1(mean, variance, alpha=1.0, beta=1.0):
    """
    Reward function to balance high mean and low variance.

    Parameters:
        mean (float): Mean of cumulative probabilities at intersection points.
        variance (float): Variance of cumulative probabilities at intersection points.
        alpha (float): Weight for maximizing the mean.
        beta (float): Weight for minimizing the variance.

    Returns:
        float: Calculated reward.
    """
    return alpha * mean - beta * variance


def reward_mean_var2(mean, variance):
    if mean > 9 and variance < 2:
        return 1
    return -1


def reward_mean_var3(mean, variance):
    if mean > 9.2 and variance < 2:
        return 2
    return -1


def reward_mean_var4(mean, variance):
    if mean > 9.2 and variance < 1.8:
        return 2
    return -1
