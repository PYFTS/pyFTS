import numpy as np


def get_data(b=0.1, c=0.2, tau=17, initial_values = np.linspace(0.5,1.5,18), iterations=1000):
    '''
    Mackey, M. C. and Glass, L. (1977). Oscillation and chaos in physiological control systems.
    Science, 197(4300):287-289.

    dy/dt = -by(t)+ cy(t - tau) / 1+y(t-tau)^10

    :param b: Equation coefficient
    :param c: Equation coefficient
    :param tau: Lag parameter, default: 17
    :param initial_values: numpy array with the initial values of y. Default: np.linspace(0.5,1.5,18)
    :param iterations: number of iterations. Default: 1000
    :return:
    '''
    y = initial_values.tolist()

    for n in np.arange(len(y), iterations+100):
        y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))

    return y[100:]