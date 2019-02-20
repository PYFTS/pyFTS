"""
Facilities to generate synthetic stochastic processes
"""

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


def generate_linear_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max, it=100, num=10, vmin=None, vmax=None):
    """

    :param period:
    :param mu_min:
    :param sigma_min:
    :param mu_max:
    :param sigma_max:
    :param it:
    :param num:
    :param vmin:
    :param vmax:
    :return:
    """

    if period > num:
        raise("The 'period' parameter must be lesser than 'it' parameter")

    mu_inc = (mu_max - mu_min)/period
    sigma_inc = (sigma_max - sigma_min) / period
    mu = mu_min
    sigma = sigma_min
    ret = []
    signal = True

    for k in np.arange(0, it):
        tmp = np.random.normal(mu, sigma, num)
        if vmin is not None:
            tmp = np.maximum(np.full(num, vmin), tmp)
        if vmax is not None:
            tmp = np.minimum(np.full(num, vmax), tmp)
        ret.extend(tmp)

        if k % period == 0:
            signal = not signal

        mu += (mu_inc if signal else -mu_inc)
        sigma += (sigma_inc if signal else -sigma_inc)

    return ret


def generate_senoidal_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max, it=100, num=10, vmin=None, vmax=None):
    """

    :param period:
    :param mu_min:
    :param sigma_min:
    :param mu_max:
    :param sigma_max:
    :param it:
    :param num:
    :param vmin:
    :param vmax:
    :return:
    """

    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    mu = mu_min
    sigma = sigma_min
    ret = []

    for k in np.arange(0, it):
        tmp = np.random.normal(mu, sigma, num)
        if vmin is not None:
            tmp = np.maximum(np.full(num, vmin), tmp)
        if vmax is not None:
            tmp = np.minimum(np.full(num, vmax), tmp)
        ret.extend(tmp)

        mu += mu_range * np.sin (period * k)
        sigma += mu_range * np.sin (period * k)

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


def _append(additive, start, before, new):
    if not additive:
        before.extend(new)
        return before
    else:
        for k in range(start):
            new.insert(0,0)

        l = len(before)

        if len(before) == 0:
            tmp = np.array(new)
        else:
            tmp = np.array(before) + np.array(new[:l])
        return tmp.tolist()



class SignalEmulator(object):

    def __init__(self, **kwargs):
        super(SignalEmulator, self).__init__()

        self.components = []

    def stationary_gaussian(self, mu, sigma, **kwargs):
        """
        Creates a continuous Gaussian signal with mean mu and variance sigma.
        :param mu: mean
        :param sigma: variance
        :keyword cummulative: If False it cancels the previous signal and start this one, if True
                              this signal is added to the previous one
        :keyword start: lag index to start this signal, the default value is 0
        :keyword it: Number of iterations, the default value is 1
        :keyword length: Number of samples generated on each iteration, the default value is 100
        :keyword vmin: Lower bound value of generated data, the default value is None
        :keyword vmax: Upper bound value of generated data, the default value is None
        :return: A list of it*num float values
        """
        parameters = {'mu': mu, 'sigma': sigma}
        self.components.append({'dist': 'gaussian', 'type': 'constant',
                                'parameters': parameters, 'args': kwargs})

    def incremental_gaussian(self, mu, sigma, **kwargs):
        """
        Creates an additive gaussian interference on a previous signal
        :param mu:
        :param sigma:
        :keyword cummulative: If False it cancels the previous signal and start this one, if True
                              this signal is added to the previous one
        :keyword start: lag index to start this signal, the default value is 0
        :keyword it: Number of iterations, the default value is 1
        :keyword length: Number of samples generated on each iteration, the default value is 100
        :keyword vmin: Lower bound value of generated data, the default value is None
        :keyword vmax: Upper bound value of generated data, the default value is None
        :return: A list of it*num float values
        """
        parameters = {'mu': mu, 'sigma': sigma}
        self.components.append({'dist': 'gaussian', 'type': 'incremental',
                                'parameters': parameters, 'args': kwargs})

    def periodic_gaussian(self, type, period, mu_min, sigma_min, mu_max, sigma_max, **kwargs):
        """
        Creates an additive periodic gaussian interference on a previous signal
        :param mu:
        :param sigma:
        :keyword additive: If False it cancels the previous signal and start this one, if True
                              this signal is added to the previous one
        :keyword start: lag index to start this signal, the default value is 0
        :keyword it: Number of iterations, the default value is 1
        :keyword length: Number of samples generated on each iteration, the default value is 100
        :keyword vmin: Lower bound value of generated data, the default value is None
        :keyword vmax: Upper bound value of generated data, the default value is None
        :return: A list of it*num float values
        """
        parameters = {'type':type, 'period':period,
                      'mu_min': mu_min, 'sigma_min': sigma_min, 'mu_max': mu_max, 'sigma_max': sigma_max}
        self.components.append({'dist': 'gaussian', 'type': 'periodic',
                                'parameters': parameters, 'args': kwargs})

    def blip(self, intensity, **kwargs):
        """

        :param intensity:
        :param kwargs:
        :return:
        """
        self.components.append({'dist': 'blip', 'type': 'blip', 'parameters': [intensity, kwargs]})

    def run(self):
        signal = []
        last_it = 10
        last_num = 10
        for ct, component in enumerate(self.components):
            parameters = component['parameters']
            kwargs = component['args']
            additive = kwargs.get('additive', True)
            start = kwargs.get('start', 0)
            it = kwargs.get('it', last_it)
            num = kwargs.get('length', last_num)
            vmin = kwargs.get('vmin',None)
            vmax = kwargs.get('vmax', None)
            if component['type'] == 'constant':
                tmp = generate_gaussian_linear(parameters['mu'], parameters['sigma'], 0, 0,
                                         it=it, num=num, vmin=vmin, vmax=vmax)
            elif component['type'] == 'incremental':
                tmp = generate_gaussian_linear(0, 0, parameters['mu'], parameters['sigma'],
                                         it=num, num=1, vmin=vmin, vmax=vmax)
            elif component['type'] == 'periodic':
                period = component['period']
                mu_min, sigma_min = parameters['mu_min'],parameters['sigma_min']
                mu_max, sigma_max = parameters['mu_max'],parameters['sigma_max']

                if parameters['type'] == 'senoidal':
                    tmp = generate_senoidal_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max,
                                                        it=num, num=1, vmin=vmin, vmax=vmax)
                else:
                    tmp = generate_linear_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max,
                                                        it=num, num=1, vmin=vmin, vmax=vmax)

            last_num = num
            last_it = it

            signal = _append(additive, start, signal, tmp)

        return signal

