import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.models.nonstationary import common, perturbation, partitioners, util
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

transformations = {'None': None, 'Differential(1)': tdiff, 'BoxCox(0)': boxcox }

from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.models import chen

partitions = {'CMIV': {'BoxCox(0)': 17, 'Differential(1)': 7, 'None': 13},
 'IMCV': {'BoxCox(0)': 22, 'Differential(1)': 9, 'None': 25},
 'IMIV': {'BoxCox(0)': 27, 'Differential(1)': 11, 'None': 6},
 'NASDAQ': {'BoxCox(0)': 39, 'Differential(1)': 10, 'None': 34},
 'SP500': {'BoxCox(0)': 38, 'Differential(1)': 15, 'None': 39},
 'TAIEX': {'BoxCox(0)': 36, 'Differential(1)': 18, 'None': 38}}


tag = 'benchmarks'

def nsfts_partitioner_builder(data, npart, transformation):
    from pyFTS.partitioners import Grid
    from pyFTS.models.nonstationary import perturbation, partitioners

    tmp_fs = Grid.GridPartitioner(data=data, npart=npart, transformation=transformation)
    fs = partitioners.SimpleNonStationaryPartitioner(data, tmp_fs,
                                                     location=perturbation.polynomial,
                                                     location_params=[1, 0],
                                                     location_roots=0,
                                                     width=perturbation.polynomial,
                                                     width_params=[1, 0],
                                                     width_roots=0)
    return fs


for ds in datasets.keys():
    dataset = datasets[ds]

    for tf in transformations.keys():
        transformation = transformations[tf]

        partitioning = partitions[ds][tf]

        bchmk.sliding_window_benchmarks(dataset, 2000, train=0.2, inc=0.2,
                                        benchmark_models=False,
                                        methods=[cvfts.ConditionalVarianceFTS],
                                        partitioners_methods=[nsfts_partitioner_builder],
                                        transformations=[transformation],
                                        partitions=[partitioning],
                                        progress=False, type='point',
                                        file="nsfts_benchmarks.db", dataset=ds, tag=tag)