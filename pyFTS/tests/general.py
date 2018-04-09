#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.common import Transformations

from pyFTS.data import TAIEX

dataset = TAIEX.get_data()

from pyFTS.benchmarks import benchmarks as bchmk

from pyFTS.models import pwfts


bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2, methods=[pwfts.ProbabilisticWeightedFTS],
                                benchmark_models=False, orders=[1], partitions=[10], #np.arange(10,100,2),
                                progress=False, type='distribution',
                                distributed=False, nodes=['192.168.0.106', '192.168.0.105', '192.168.0.110'],
                                save=True, file="pwfts_taiex_interval.csv")