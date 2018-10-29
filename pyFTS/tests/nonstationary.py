import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.models.nonstationary import common, perturbation, partitioners as nspart, util
from pyFTS.models.nonstationary import nsfts, cvfts
from pyFTS.partitioners import Grid, Entropy
import matplotlib.pyplot as plt
from pyFTS.common import Util as cUtil
import pandas as pd

from pyFTS.data import TAIEX, NASDAQ, SP500, artificial, mackey_glass

#mackey_glass.get_data()

datasets = {
    "TAIEX": TAIEX.get_data()[:4000],
    "SP500": SP500.get_data()[10000:14000],
    "NASDAQ": NASDAQ.get_data()[:4000],
    # Incremental Mean and Incremental Variance
    "IMIV": artificial.generate_gaussian_linear(1,0.2,0.2,0.05,it=100, num=40),
    # Incremental Mean and Incremental Variance, lower bound equals to 0
    "IMIV0": artificial.generate_gaussian_linear(1,0.2,0.,0.05, vmin=0,it=100, num=40),
    # Constant Mean and Incremental Variance
    "CMIV": artificial.generate_gaussian_linear(5,0.1,0,0.02,it=100, num=40),
    # Incremental Mean and Constant Variance
    "IMCV": artificial.generate_gaussian_linear(1,0.6,0.1,0,it=100, num=40)
}

train_split = 2000
test_length = 200

from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)

boxcox = Transformations.BoxCox(0)

transformations = {
    'None': None,
    'Differential(1)': tdiff,
    'BoxCox(0)': boxcox
}

from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.models import chen, hofts, pwfts, hwang

partitions = {'CMIV': {'BoxCox(0)': 36, 'Differential(1)': 11, 'None': 8},
 'IMCV': {'BoxCox(0)': 36, 'Differential(1)': 20, 'None': 16},
 'IMIV': {'BoxCox(0)': 39, 'Differential(1)': 12, 'None': 6},
 'IMIV0': {'BoxCox(0)': 39, 'Differential(1)': 12, 'None': 3},
 'NASDAQ': {'BoxCox(0)': 39, 'Differential(1)': 13, 'None': 36},
 'SP500': {'BoxCox(0)': 33, 'Differential(1)': 7, 'None': 33},
 'TAIEX': {'BoxCox(0)': 39, 'Differential(1)': 31, 'None': 33}}

from pyFTS.models.nonstationary import partitioners as nspart, cvfts, util as nsUtil
'''
#fs = nspart.simplenonstationary_gridpartitioner_builder(data=datasets['SP500'][:300],
#                                                        npart=partitions['SP500']['None'],
#                                                        transformation=None)
fs = Grid.GridPartitioner(data=datasets['SP500'][:300],
                                                        npart=15,
                                                        transformation=None)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[15, 5])

fs.plot(axes)

from pyFTS.common import Util

Util.show_and_save_image(fig, "fig2.png", True)

#nsUtil.plot_sets(fs)


'''
def model_details(ds, tf, train_split, test_split):
    data = datasets[ds]
    train = data[:train_split]
    test = data[train_split:test_split]
    transformation = transformations[tf]
    fs = nspart.simplenonstationary_gridpartitioner_builder(data=train, npart=15, #partitions[ds][tf],
                                                            transformation=transformation)
    model = nsfts.NonStationaryFTS(partitioner=fs)
    model.fit(train)
    print(model)
    forecasts = model.predict(test)
    residuals = np.array(test[1:]) - np.array(forecasts[:-1])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=[15, 10])

    axes[0].plot(test[1:], label="Original")
    axes[0].plot(forecasts[:-1], label="Forecasts")
    axes[0].set_ylabel("Univ. of Discourse")

    #axes[1].set_title("Residuals")
    axes[1].plot(residuals)
    axes[1].set_ylabel("Error")
    handles0, labels0 = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles0, labels0, loc=2)

    nsUtil.plot_sets_conditional(model, test, step=10, size=[10, 7],
                                 save=True,file="fig.png", axes=axes[2], fig=fig)

model_details('SP500','None',200,400)
#'''
print("ts")
'''
tag = 'benchmarks'


for ds in datasets.keys():
    dataset = datasets[ds]

    for tf in transformations.keys():
        transformation = transformations[tf]

        partitioning = partitions[ds][tf]

        bchmk.sliding_window_benchmarks(dataset, 3000, train=0.1, inc=0.1,
                                        #methods=[
                                        #    hwang.HighOrderFTS,
                                        #    hofts.HighOrderFTS,
                                        #    pwfts.ProbabilisticWeightedFTS],
                                        #orders = [3],
                                        benchmark_models=False,
                                        transformations=[transformation],
                                        partitions=[partitioning],
                                        progress=False, type='point',
                                        file="nsfts_benchmarks.db", dataset=ds, tag=tag)

train_split = 200
test_split = 2000
for ds in datasets.keys():
    dataset = datasets[ds]

    print(ds)

    for tf in ['None']: #transformations.keys():
        transformation = transformations[tf]
        train = dataset[:train_split]
        test = dataset[train_split:test_split]

        fs = nspart.simplenonstationary_gridpartitioner_builder(data=train,
                                                                npart=partitions[ds][tf],
                                                                transformation=transformation)
        print(fs)
        #cvfts1 = cvfts.ConditionalVarianceFTS(partitioner=fs)
        model = nsfts.NonStationaryFTS(partitioner=fs)
        model.fit(train)
        print(model)

        forecasts = model.predict(test)
    
        #print(forecasts)
  
        partitioning = partitions[ds][tf]

        bchmk.sliding_window_benchmarks(dataset, 2000, train=0.2, inc=0.2,
                                        benchmark_models=False,
                                        methods=[cvfts.ConditionalVarianceFTS],
                                        partitioners_methods=[nspart.simplenonstationary_gridpartitioner_builder],
                                        transformations=[transformation],
                                        partitions=[partitioning],
                                        progress=False, type='point',
                                        file="nsfts_benchmarks.db", dataset=ds, tag=tag)
'''
