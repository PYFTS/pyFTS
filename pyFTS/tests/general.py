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
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts
from pyFTS.common import Transformations, Membership

from pyFTS.data import artificial

"""
cd = artificial.SignalEmulator()\
    .stationary_gaussian(0,.2,length=10, it=1)\
    .incremental_gaussian(0.5, 0,start=5,length=5)\
    .blip()\
    .stationary_gaussian(3,.2,length=10, it=1, additive=False)
print([round(k,3) for k in cd.run()])
"""

signal = artificial.SignalEmulator()\
    .stationary_gaussian(1,0.2,length=130,it=10)\
    .periodic_gaussian('sinoidal',100, 0.5,0.5,10,1,start=100,length=2000)\
    .blip()\
    .run()
