import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.models.nonstationary import common, perturbation, partitioners, util
from pyFTS.models.nonstationary import nsfts, cvfts
from pyFTS.partitioners import Grid, Entropy
import matplotlib.pyplot as plt
from pyFTS.common import Util as cUtil
import pandas as pd

'''
from pyFTS.data import artificial

lmv1 = artificial.generate_gaussian_linear(1,0.2,0.2,0.05)

ts=200
ws=35
train1 = lmv1[:ts]
test1 = lmv1[ts:]

tmp_fs1 = Grid.GridPartitioner(data=train1[:50], npart=10)

fs1 = partitioners.PolynomialNonStationaryPartitioner(train1, tmp_fs1, window_size=ws, degree=1)

nsfts1 = honsfts.HighOrderNonStationaryFTS("", partitioner=fs1)

nsfts1.fit(train1, order=2, parameters=ws)

print(fs1)

print(nsfts1.predict(test1))

print(nsfts1)

util.plot_sets(fs1, tam=[10, 5], start=0, end=100, step=2, data=lmv1[:100], window_size=35)
'''


from pyFTS.common import Transformations
tdiff = Transformations.Differential(1)


from pyFTS.common import Util

from pyFTS.data import TAIEX

taiex = TAIEX.get_data()
#taiex_diff = tdiff.apply(taiex)

train = taiex[:600]
test = taiex[600:800]

#fs_tmp = Grid.GridPartitioner(data=train, npart=7, transformation=tdiff)
#fs_tmp = Entropy.EntropyPartitioner(data=train, npart=7, transformation=tdiff)
fs_tmp = Grid.GridPartitioner(data=train, npart=20)

fs = partitioners.SimpleNonStationaryPartitioner(train, fs_tmp)

print(fs)

model = cvfts.ConditionalVarianceFTS(partitioner=fs,memory_window=3)
model.fit(train)

print(model)

#tmpp4 = model.predict(test, type='point')
#tmp = model.predict(test, type='interval')

#util.plot_sets_conditional(model, tdiff.apply(test),  step=5, size=[10, 5])
#util.plot_sets_conditional(model, test,  step=5, size=[10, 5])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[10, 5])

axes[0].plot(test[1:], label="Test Data")

forecasts = model.predict(test, type='point')

axes[0].plot(forecasts[:-1], label="CVFTS Forecasts")

handles0, labels0 = axes[0].get_legend_handles_labels()
lgd = axes[0].legend(handles0, labels0, loc=2)

residuals = np.array(test[1:]) - np.array(forecasts[:-1])

axes[1].plot(residuals)
axes[1].set_title("Residuals")

print("fim")