import os
import numpy as np
from pyFTS.common import Membership
from pyFTS.nonstationary import common,perturbation,util,nsfts, honsfts
from pyFTS.partitioners import Grid
import matplotlib.pyplot as plt
import pandas as pd
os.chdir("/home/petronio/Dropbox/Doutorado/Codigos/")

"""
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
#lmv1 = generate_heteroskedastic_linear(5,0.1,0,0.2)
#lmv1 = generate_heteroskedastic_linear(1,0.3,1,0)

ns = 5 #number of fuzzy sets
ts = 200
train = lmv1[:ts]
test = lmv1[ts:]
w = 25
deg = 4

tmp_fs = Grid.GridPartitioner(train[:35], 10)

fs = common.PolynomialNonStationaryPartitioner(train, tmp_fs, window_size=35, degree=1)

nsfts1 = nsfts.NonStationaryFTS("", partitioner=fs)

nsfts1.train(train[:100])

print(fs)

print(nsfts1)

tmp = nsfts1.forecast(test[:10], time_displacement=200)

print(tmp)
"""

passengers = pd.read_csv("DataSets/AirPassengers.csv", sep=",")
passengers = np.array(passengers["Passengers"])

ts = 100
ws=12

trainp = passengers[:ts]
testp = passengers[ts:]

tmp_fsp = Grid.GridPartitioner(trainp[:ws], 15)

fsp = common.PolynomialNonStationaryPartitioner(trainp, tmp_fsp, window_size=ws, degree=1)


#nsftsp = honsfts.HighOrderNonStationaryFTS("", partitioner=fsp)
nsftsp = nsfts.NonStationaryFTS("", partitioner=fsp, method='fuzzy')

#nsftsp.train(trainp, order=1, parameters=ws)

print(fsp)

#print(nsftsp)

#tmpp = nsftsp.forecast(passengers[55:65], time_displacement=55, window_size=ws)

#print(passengers[100:120])
#print(tmpp)

#util.plot_sets(fsp.sets,tam=[10, 5], start=0, end=100, step=2, data=passengers[:100],
#               window_size=ws, only_lines=False)

#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[15,5])

"""
axes.plot(testp, label="Original")
#axes.plot(tmpp, label="NSFTS")

handles0, labels0 = axes.get_legend_handles_labels()
lgd = axes.legend(handles0, labels0, loc=2)
"""