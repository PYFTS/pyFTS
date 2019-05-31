#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from pyFTS.common import Util as cUtil, FuzzySet
from pyFTS.partitioners import Grid, Entropy, Util as pUtil, Simple
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts, tsaur, song, sadaei, ifts
from pyFTS.models.ensemble import ensemble
from pyFTS.common import Transformations, Membership
from pyFTS.benchmarks import arima, quantreg, BSTS, gaussianproc
from pyFTS.fcm import fts, common, GA

from pyFTS.data import TAIEX, NASDAQ, SP500

datasets = {}

datasets['TAIEX'] = TAIEX.get_data()[:5000]
datasets['NASDAQ'] = NASDAQ.get_data()[:5000]
datasets['SP500'] = SP500.get_data()[10000:15000]

methods = [ensemble.SimpleEnsembleFTS]*8

methods_parameters = [
    {'name': 'EnsembleFTS-HOFTS-10-.05', 'fts_method': hofts.HighOrderFTS, 'partitions': np.arange(20,50,10), 'alpha': .05},
    {'name': 'EnsembleFTS-HOFTS-5-.05', 'fts_method': hofts.HighOrderFTS, 'partitions': np.arange(20,50,5), 'alpha': .05},
    {'name': 'EnsembleFTS-HOFTS-10-.25', 'fts_method': hofts.HighOrderFTS, 'partitions': np.arange(20,50,10), 'alpha': .25},
    {'name': 'EnsembleFTS-HOFTS-5-.25', 'fts_method': hofts.HighOrderFTS, 'partitions': np.arange(20,50,5), 'alpha': .25},
    {'name': 'EnsembleFTS-WHOFTS-10-.05', 'fts_method': hofts.WeightedHighOrderFTS, 'partitions': np.arange(20,50,10), 'alpha': .05},
    {'name': 'EnsembleFTS-WHOFTS-5-.05', 'fts_method': hofts.WeightedHighOrderFTS, 'partitions': np.arange(20,50,5), 'alpha': .05},
    {'name': 'EnsembleFTS-WHOFTS-10-.25', 'fts_method': hofts.WeightedHighOrderFTS, 'partitions': np.arange(20,50,10), 'alpha': .25},
    {'name': 'EnsembleFTS-WHOFTS-5-.25', 'fts_method': hofts.WeightedHighOrderFTS, 'partitions': np.arange(20,50,5), 'alpha': .25},
]

for dataset_name, dataset in datasets.items():
    bchmk.sliding_window_benchmarks2(dataset, 1000, train=0.8, inc=0.2,
                                    methods=methods,
                                    methods_parameters=methods_parameters,
                                    benchmark_models=False,
                                    transformations=[None],
                                    orders=[3],
                                    partitions=[None],
                                    type='interval',
                                    #distributed=True, nodes=['192.168.0.110', '192.168.0.107','192.168.0.106'],
                                    file="tmp.db", dataset=dataset_name, tag="gridsearch")
