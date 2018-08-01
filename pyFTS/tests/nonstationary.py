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

tag = 'chen_partitioning'

for ds in ['IMIV0']: #datasets.keys():
    dataset = datasets[ds]

    bchmk.sliding_window_benchmarks(dataset, 4000, train=0.2, inc=0.2,
                                    methods=[chen.ConventionalFTS],
                                    benchmark_models=False,
                                    transformations=[boxcox], #transformations[t] for t in transformations.keys()],
                                    partitions=np.arange(3, 40, 1),
                                    progress=False, type='point',
                                    file="nsfts_benchmarks.db", dataset=ds, tag=tag)