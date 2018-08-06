import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.models.nonstationary import common, perturbation, partitioners as nspart, util
from pyFTS.models.nonstationary import nsfts, cvfts
from pyFTS.partitioners import Grid, Entropy
import matplotlib.pyplot as plt
from pyFTS.common import Util as cUtil
import pandas as pd

from pyFTS.data import TAIEX, NASDAQ, SP500, artificial

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


tag = 'benchmarks'
'''
for ds in datasets.keys():
    dataset = datasets[ds]

    for tf in transformations.keys():
        transformation = transformations[tf]

        partitioning = partitions[ds][tf]

        bchmk.sliding_window_benchmarks(dataset, 2000, train=0.2, inc=0.2,
                                        methods=[
                                            hwang.HighOrderFTS,
                                            hofts.HighOrderFTS,
                                            pwfts.ProbabilisticWeightedFTS],
                                        #orders = [3],
                                        benchmark_models=False,
                                        transformations=[transformation],
                                        partitions=[partitioning],
                                        progress=False, type='point',
                                        file="nsfts_benchmarks.db", dataset=ds, tag=tag)
'''
train_split = 200
test_split = 2000
for ds in datasets.keys():
    dataset = datasets[ds]

    print(ds)

    for tf in ['None']: #transformations.keys():
        transformation = transformations[tf]
        train = dataset[:train_split]
        test = dataset[train_split:test_split]

        fs = nspart.simplenonstationary_gridpartitioner_builder(data=train, npart=partitions[ds][tf], transformation=transformation)
        print(fs)
        #cvfts1 = cvfts.ConditionalVarianceFTS(partitioner=fs)
        model = nsfts.NonStationaryFTS(partitioner=fs)
        model.fit(train)
        print(model)

        forecasts = model.predict(test)
        '''
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
