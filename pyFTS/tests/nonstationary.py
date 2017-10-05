import numpy as np
from pyFTS.common import Membership
from pyFTS.nonstationary import common,perturbation,util
from pyFTS.partitioners import Grid


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
w = 25
deg = 4


tmp_fs = Grid.GridPartitioner(train[:35], 10)

fs = common.PolynomialNonStationaryPartitioner(train, tmp_fs, window_size=35, degree=1)

uod = np.arange(0, 2, step=0.02)

util.plot_sets(uod, fs.sets,tam=[15, 5], start=0, end=10)

for set in fs.sets:
    print(set)
