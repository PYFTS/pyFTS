import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.nonstationary import common,perturbation, partitioners, util, honsfts, cvfts
from pyFTS.models.nonstationary import nsfts
from pyFTS.partitioners import Grid
import matplotlib.pyplot as plt
from pyFTS.common import Util as cUtil
import pandas as pd
os.chdir("/home/petronio/Dropbox/Doutorado/Codigos/")

data = pd.read_csv("DataSets/synthetic_nonstationary_dataset_A.csv", sep=";")
data = np.array(data["0"][:])

for ct, train, test in cUtil.sliding_window(data, 300):
    for partition in np.arange(10,50):
        print(partition)
        tmp_fsp = Grid.GridPartitioner(train, partition)
        print(len(tmp_fsp.sets))

        fsp = partitioners.PolynomialNonStationaryPartitioner(train, tmp_fsp, window_size=35, degree=1)

'''
diff = Transformations.Differential(1)

def generate_heteroskedastic_linear(mu_ini, sigma_ini, mu_inc, sigma_inc, it=10, num=35):
    mu = mu_ini
    sigma = sigma_ini
    ret = []
    for k in np.arange(0,it):
        ret.extend(np.random.normal(mu, sigma, num))
        mu += mu_inc
        sigma += sigma_inc
    return ret


#lmv1 = generate_heteroskedastic_linear(1,0.1,1,0.3)
lmv1 = generate_heteroskedastic_linear(5,0.1,0,0.2)
#lmv1 = generate_heteroskedastic_linear(1,0.3,1,0)

lmv1 = diff.apply(lmv1)

ns = 10 #number of fuzzy sets
ts = 200
train = lmv1[:ts]
test = lmv1[ts:]
w = 25
deg = 4

tmp_fs = Grid.GridPartitioner(train, 10)

#fs = partitioners.PolynomialNonStationaryPartitioner(train, tmp_fs, window_size=35, degree=1)
fs = partitioners.ConstantNonStationaryPartitioner(train, tmp_fs,
                                                   location=perturbation.polynomial,
                                                   location_params=[1,0],
                                                   location_roots=0,
                                                   width=perturbation.polynomial,
                                                   width_params=[1,0],
                                                   width_roots=0)
'''
"""
perturb = [0.5, 0.25]
for i in [0,1]:
    print(fs.sets[i].parameters)
    fs.sets[i].perturbate_parameters(perturb[i])
for i in [0,1]:
    print(fs.sets[i].perturbated_parameters[perturb[i]])
"""
'''
#nsfts1 = nsfts.NonStationaryFTS("", partitioner=fs)

nsfts1 = cvfts.ConditionalVarianceFTS("", partitioner=fs)

nsfts1.train(train)

#print(fs)

#print(nsfts1)

#tmp = nsfts1.forecast(test[50:60])

#print(tmp)
#print(test[50:60])

util.plot_sets_conditional(nsfts1, test, end=150, step=1,tam=[10, 5])
print('')
"""
passengers = pd.read_csv("DataSets/AirPassengers.csv", sep=",")
passengers = np.array(passengers["Passengers"])

ts = 100
ws=12

trainp = passengers[:ts]
testp = passengers[ts:]

tmp_fsp = Grid.GridPartitioner(trainp[:50], 10)


fsp = common.PolynomialNonStationaryPartitioner(trainp, tmp_fsp, window_size=ws, degree=1)


nsftsp = honsfts.HighOrderNonStationaryFTS("", partitioner=fsp)
#nsftsp = nsfts.NonStationaryFTS("", partitioner=fsp, method='fuzzy')

nsftsp.train(trainp, order=2, parameters=ws)

#print(fsp)

#print(nsftsp)

tmpp = nsftsp.forecast(passengers[101:104], time_displacement=101, window_size=ws)
tmpi = nsftsp.forecast_interval(passengers[101:104], time_displacement=101, window_size=ws)

#print(passengers[101:104])
print([k[0] for k in tmpi])
print(tmpp)
print([k[1] for k in tmpi])

#util.plot_sets(fsp.sets,tam=[10, 5], start=0, end=100, step=2, data=passengers[:100],
#               window_size=ws, only_lines=False)

#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[15,5])
"""

"""
axes.plot(testp, label="Original")
#axes.plot(tmpp, label="NSFTS")

handles0, labels0 = axes.get_legend_handles_labels()
lgd = axes.legend(handles0, labels0, loc=2)
"""
'''