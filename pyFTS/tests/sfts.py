#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

import pandas as pd
from pyFTS.partitioners import Grid, CMeans, FCM, Entropy
from pyFTS.common import FLR,FuzzySet,Membership,Transformations,Util
from pyFTS import fts
from pyFTS.seasonal import sfts, msfts
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import Measures

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

sonda = pd.read_csv("DataSets/SONDA_BSB_MOD.csv", sep=";")

sonda['data'] = pd.to_datetime(sonda['data'])

sonda = sonda[:][527041:]

sonda.index = np.arange(0,len(sonda.index))


sonda_treino = sonda[:1051200]
sonda_teste = sonda[1051201:]


#res = bchmk.simpleSearch_RMSE(sonda_treino, sonda_teste,
#                              sfts.SeasonalFTS,np.arange(3,30),[1],parameters=1440,
#                              tam=[15,8], plotforecasts=False,elev=45, azim=40,
#                               save=False,file="pictures/sonda_sfts_error_surface", intervals=False)

from pyFTS.models.seasonal import SeasonalIndexer
from pyFTS.models import msfts
from pyFTS.common import FLR

partitions = ['grid','entropy']

indexers = ['m15','Mh','Mhm15']

models = []
ixs = []

sample = sonda_teste[0:4300]

for max_part in [10, 20, 30, 40, 50]:
    for part in partitions:
        for ind in indexers:
            ix = Util.load_obj("models/sonda_ix_" + ind + ".pkl")
            model = Util.load_obj("models/sonda_msfts_" + part + "_" + str(max_part) + "_" + ind + ".pkl")
            model.shortname = part + "_" + str(max_part) + "_" + ind

            models.append(model)
            ixs.append(ix)

print(bchmk.print_point_statistics(sample, models, indexers=ixs))