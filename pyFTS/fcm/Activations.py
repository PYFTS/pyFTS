import numpy as np


def step(x):
    if x <= 0:
        return 0
    else:
        return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
  mvs = sum([np.exp(k) for k in x.flatten()])
  return np.array([np.exp(k)/mvs for k in x.flatten()])

