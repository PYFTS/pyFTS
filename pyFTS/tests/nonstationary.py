import numpy as np
from pyFTS.common import Membership
from pyFTS.nonstationary import common,perturbation,util,nsfts
from pyFTS.partitioners import Grid
import matplotlib.pyplot as plt


def generate_heteroskedastic_linear(mu_ini, sigma_ini, mu_inc, sigma_inc, it=10, num=35):
    mu = mu_ini
    sigma = sigma_ini
    ret = []
    for k in np.arange(0,it):
        ret.extend(np.random.normal(mu, sigma, num))
        mu += mu_inc
        sigma += sigma_inc
    return ret


lmv1 = generate_heteroskedastic_linear(1,0.1,1,0.3)

ns = 5 #number of fuzzy sets
ts = 200
train = lmv1[:ts]
test = lmv1[ts:]
w = 25
deg = 4

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,5])

tmp_fs = Grid.GridPartitioner(train[:35], 10)

fs = common.PolynomialNonStationaryPartitioner(train, tmp_fs, window_size=35, degree=1)

nsfts1 = nsfts.NonStationaryFTS("", partitioner=fs)

nsfts1.train(train[:35])

tmp = nsfts1.forecast(test, time_displacement=200)

axes.plot(test)
axes.plot(tmp)

print(tmp)
