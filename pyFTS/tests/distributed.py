from pyFTS.partitioners import Grid
from pyFTS.models import chen
from pyFTS.benchmarks import Measures
from pyFTS.common import Util as cUtil, fts
import pandas as pd
import numpy as np
import os
from pyFTS.common import Transformations
from copy import deepcopy
from pyFTS.models import pwfts
from pyFTS.benchmarks import benchmarks as bchmk, Measures

import time

from pyFTS.data import SONDA, Malaysia

datasets = {}

sonda = SONDA.get_dataframe()[['datahora','glo_avg','ws_10m']]

sonda = sonda.drop(sonda.index[np.where(sonda["ws_10m"] <= 0.01)])
sonda = sonda.drop(sonda.index[np.where(sonda["glo_avg"] <= 0.01)])
sonda = sonda.dropna()

malaysia = Malaysia.get_dataframe()

datasets['SONDA.ws_10m'] = sonda["ws_10m"].values
datasets['SONDA.glo_avg'] = sonda["glo_avg"].values
datasets['Malaysia.temperature'] = malaysia["temperature"].values
datasets['Malaysia.load'] = malaysia["load"].values

windows = [600000, 600000, 10000, 10000]

cpus = 3

for ct, (dataset_name, dataset) in enumerate(datasets.items()):
    bchmk.train_test_time(dataset, windowsize=windows[ct], train=0.9, inc=.5,
                     methods=[pwfts.ProbabilisticWeightedFTS],
                     order=2,
                     partitions=50,
                     steps=cpus,
                     num_batches=cpus,
                     distributed='dispy', nodes=['192.168.0.110'], #, '192.168.0.107','192.168.0.106'],
                     file="experiments.db", dataset=dataset_name,
                     tag="speedup")
