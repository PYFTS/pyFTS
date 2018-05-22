import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.models.nonstationary import common, perturbation, partitioners, util
from pyFTS.models.nonstationary import nsfts, cvfts
from pyFTS.partitioners import Grid
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
taiex_diff = tdiff.apply(taiex)

train = taiex_diff[:600]
test = taiex_diff[600:1500]

fs_tmp = Grid.GridPartitioner(data=train, npart=20) #, transformation=tdiff)

fs = partitioners.SimpleNonStationaryPartitioner(train, fs_tmp)

print(fs)

model = cvfts.ConditionalVarianceFTS(partitioner=fs)
model.fit(train)

print(model)

#tmpp4 = model.predict(test, type='point')
tmp = model.predict(test, type='interval')

#util.plot_sets_conditional(model, test,  step=1, tam=[10, 5])