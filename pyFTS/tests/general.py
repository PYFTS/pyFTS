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

from pyFTS.data import TAIEX

fs = Grid.GridPartitioner(data=TAIEX.get_data(), npart=23)

print(fs.min, fs.max)

tmp = fs.search(5500)
print(tmp)

tmp = fs.fuzzyfy(5500, method='fuzzy', alpha_cut=0.3)
print(tmp)


