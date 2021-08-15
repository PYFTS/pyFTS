"""
Facilities to generate synthetic stochastic processes
"""

import numpy as np


class SignalEmulator(object):
    """
    Emulate a complex signal built from several additive and non-additive components
    """

    def __init__(self, **kwargs):
        super(SignalEmulator, self).__init__()

        self.components = []
        """Components of the signal"""

    def stationary_gaussian(self, mu:float, sigma:float, **kwargs):
        """
        Creates a continuous Gaussian signal with mean mu and variance sigma.

        :param mu: mean
        :param sigma: variance
        :keyword additive: If False it cancels the previous signal and start this one, if True
                           this signal is added to the previous one
        :keyword start: lag index to start this signal, the default value is 0
        :keyword it: Number of iterations, the default value is 1
        :keyword length: Number of samples generated on each iteration, the default value is 100
        :keyword vmin: Lower bound value of generated data, the default value is None
        :keyword vmax: Upper bound value of generated data, the default value is None
        :return: the current SignalEmulator instance, for method chaining
        """
        parameters = {'mu': mu, 'sigma': sigma}
        self.components.append({'dist': 'gaussian', 'type': 'constant',
                                'parameters': parameters, 'args': kwargs})
        return self

    def incremental_gaussian(self, mu:float, sigma:float, **kwargs):
        """
        Creates an additive gaussian interference on a previous signal

        :param mu: increment on mean
        :param sigma: increment on variance
        :keyword start: lag index to start this signal, the default value is 0
        :keyword it: Number of iterations, the default value is 1
        :keyword length: Number of samples generated on each iteration, the default value is 100
        :keyword vmin: Lower bound value of generated data, the default value is None
        :keyword vmax: Upper bound value of generated data, the default value is None
        :return: the current SignalEmulator instance, for method chaining
        """
        parameters = {'mu': mu, 'sigma': sigma}
        self.components.append({'dist': 'gaussian', 'type': 'incremental',
                                'parameters': parameters, 'args': kwargs})
        return self

    def periodic_gaussian(self, type, period, mu_min, sigma_min, mu_max, sigma_max, **kwargs):
        """
        Creates an additive periodic gaussian interference on a previous signal

        :param type: 'linear' or 'sinoidal'
        :param period: the period of recurrence
        :param mu: increment on mean
        :param sigma: increment on variance
        :keyword start: lag index to start this signal, the default value is 0
        :keyword it: Number of iterations, the default value is 1
        :keyword length: Number of samples generated on each iteration, the default value is 100
        :keyword vmin: Lower bound value of generated data, the default value is None
        :keyword vmax: Upper bound value of generated data, the default value is None
        :return: the current SignalEmulator instance, for method chaining
        """
        parameters = {'type':type, 'period':period,
                      'mu_min': mu_min, 'sigma_min': sigma_min, 'mu_max': mu_max, 'sigma_max': sigma_max}
        self.components.append({'dist': 'gaussian', 'type': 'periodic',
                                'parameters': parameters, 'args': kwargs})
        return self

    def blip(self, **kwargs):
        """
        Creates an outlier greater than the maximum or lower then the minimum previous values of the signal,
        and insert it on a random location of the signal.

        :return: the current SignalEmulator instance, for method chaining
        """
        parameters = {}
        self.components.append({'dist': 'blip', 'type': 'blip',
                                'parameters': parameters, 'args':kwargs})
        return self

    def run(self):
        """
        Render the signal

        :return: a list of float values
        """
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
                period = parameters['period']
                mu_min, sigma_min = parameters['mu_min'],parameters['sigma_min']
                mu_max, sigma_max = parameters['mu_max'],parameters['sigma_max']

                if parameters['type'] == 'sinoidal':
                    tmp = generate_sinoidal_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max,
                                                              it=num, num=1, vmin=vmin, vmax=vmax)
                else:
                    tmp = generate_linear_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max,
                                                        it=num, num=1, vmin=vmin, vmax=vmax)
            elif component['type'] == 'blip':
                _mx = np.nanmax(signal)
                _mn = np.nanmin(signal)

                _mx += 2*_mx if _mx > 0 else -2*_mx
                _mn += -2*_mn if _mn > 0 else 2*_mn

                if vmax is not None:
                    _mx = min(_mx, vmax) if vmax > 0 else max(_mx, vmax)
                if vmin is not None:
                    _mn = max(_mn, vmin) if vmin > 0 else min(_mn, vmin)

                start = np.random.randint(0, len(signal))
                tmp = [_mx] if np.random.rand() >= .5 else [-_mn]

            last_num = num
            last_it = it

            signal = _append(additive, start, signal, tmp)

        return signal




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
    Generates a periodic linear variation on mean and variance

    :param period: the period of recurrence
    :param mu_min: initial (and minimum) mean of each period
    :param sigma_min: initial (and minimum) variance of each period
    :param mu_max: final (and maximum) mean of each period
    :param sigma_max: final (and maximum) variance of each period
    :param it: Number of iterations
    :param num: Number of samples generated on each iteration
    :param vmin: Lower bound value of generated data
    :param vmax: Upper bound value of generated data
    :return: A list of it*num float values
    """

    if period > it:
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

        sigma = max(sigma, 0.005)

    return ret


def generate_sinoidal_periodic_gaussian(period, mu_min, sigma_min, mu_max, sigma_max, it=100, num=10, vmin=None, vmax=None):
    """
    Generates a periodic sinoidal variation on mean and variance

    :param period: the period of recurrence
    :param mu_min: initial (and minimum) mean of each period
    :param sigma_min: initial (and minimum) variance of each period
    :param mu_max: final (and maximum) mean of each period
    :param sigma_max: final (and maximum) variance of each period
    :param it: Number of iterations
    :param num: Number of samples generated on each iteration
    :param vmin: Lower bound value of generated data
    :param vmax: Upper bound value of generated data
    :return: A list of it*num float values
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

        mu += mu_range * np.sin(period * k)
        sigma += sigma_range * np.sin(period * k)

        sigma = max(sigma, 0.005)

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
    """
    Simple Gaussian noise signal
    :param n: number of samples
    :return:
    """
    return np.random.normal(0, 1, n)


def random_walk(n=500, type='gaussian'):
    """
    Simple random walk

    :param n: number of samples
    :param type: 'gaussian' or 'uniform'
    :return:
    """
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

        l1 = len(before)
        l2 = len(new)

        if l2 < l1:
            new.extend(np.zeros(l1 - l2).tolist())
        elif 0 < l1 < l2:
            new = new[:l1]

        if len(before) == 0:
            tmp = np.array(new)
        else:
            tmp = np.array(before) + np.array(new)
        return tmp.tolist()


