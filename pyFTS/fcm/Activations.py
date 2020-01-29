import numpy as np


def step(x, deriv=False):
    if deriv:
        1 * (x == 0)
    else:
        return 1 * (x > 0)


def sigmoid(x, deriv=False):
    if deriv:
        #return sigmoid(x)*(1 - sigmoid(x))
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, deriv=False):
    if deriv:
        pass
    else:
        mvs = sum([np.exp(k) for k in x.flatten()])
        return np.array([np.exp(k)/mvs for k in x.flatten()])


def tanh(x, deriv=False):
    if deriv:
        pass
    else:
        return np.tanh(x)


def relu(x, deriv=False):
    if deriv:
        return 1. * (x > 0)
    else:
        return x * (x > 0)

