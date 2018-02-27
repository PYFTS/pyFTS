import numpy as np


def generate_gaussian_linear(mu_ini, sigma_ini, mu_inc, sigma_inc, it=100, num=10, vmin=None, vmax=None):
    """
    Generate data sampled from Gaussian distribution, with constant or linear changing parameters
    :param mu_ini: Initial mean
    :param sigma_ini: Initial variance
    :param mu_inc:  Mean increment after 'num' samples
    :param sigma_inc: Variance increment after 'num' samples
    :param it: Number of iterations
    :param num: Number of samples generated on each iteration
    :param vmin: Lower bound value of generated data
    :param vmax: Upper bound value of generated data
    :return: A list of it*num float values
    """
    mu = mu_ini
    sigma = sigma_ini
    ret = []
    for k in np.arange(0,it):
        tmp = np.random.normal(mu, sigma, num)
        if vmin is not None:
            tmp = np.maximum(np.full(num, vmin), tmp)
        if vmax is not None:
            tmp = np.minimum(np.full(num, vmax), tmp)
        ret.extend(tmp)
        mu += mu_inc
        sigma += sigma_inc
    return ret


def generate_uniform_linear(min_ini, max_ini, min_inc, max_inc, it=100, num=10, vmin=None, vmax=None):
    """
    Generate data sampled from Uniform distribution, with constant or  linear changing bounds
    :param mu_ini: Initial mean
    :param sigma_ini: Initial variance
    :param mu_inc:  Mean increment after 'num' samples
    :param sigma_inc: Variance increment after 'num' samples
    :param it: Number of iterations
    :param num: Number of samples generated on each iteration
    :param vmin: Lower bound value of generated data
    :param vmax: Upper bound value of generated data
    :return: A list of it*num float values
    """
    _min = min_ini
    _max = max_ini
    ret = []
    for k in np.arange(0,it):
        tmp = np.random.uniform(_min, _max, num)
        if vmin is not None:
            tmp = np.maximum(np.full(num, vmin), tmp)
        if vmax is not None:
            tmp = np.minimum(np.full(num, vmax), tmp)
        ret.extend(tmp)
        _min += min_inc
        _max += max_inc
    return ret


def white_noise(n=500):
    return np.random.normal(0, 1, n)


def random_walk(n=500, type='gaussian'):
    if type == 'gaussian':
        tmp = generate_gaussian_linear(0, 1, 0, 0, it=1, num=n)
    else:
        tmp = generate_uniform_linear(-1, 1, 0, 0, it=1, num=n)
    ret = [0]
    for i in range(n):
        ret.append(tmp[i] + ret[i])

    return ret

