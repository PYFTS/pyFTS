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

cd = artificial.SignalEmulator()
cd.stationary_gaussian(1,.2,length=10, it=1)
cd.incremental_gaussian(0.5, 0,start=5,length=5)
#cd.stationary_gaussian(3,.2,length=10, it=1, additive=True)
print(len(cd.run()))

