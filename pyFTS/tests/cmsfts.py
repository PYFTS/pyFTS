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
from pyFTS.common import FLR, FuzzySet, Membership, Transformations, Util, fts
from pyFTS import sfts
from pyFTS.models import msfts
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import Measures

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

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

from pyFTS.common import Util
from pyFTS.models import cmsfts

partitions = ['grid', 'entropy']

indexers = ['m15', 'Mh', 'Mhm15']

for max_part in [40, 50]:
    for part in partitions:
        fs = Util.load_obj("models/sonda_fs_" + part + "_" + str(max_part) + ".pkl")

        for ind in indexers:
            ix = Util.load_obj("models/sonda_ix_" + ind + ".pkl")

            model = cmsfts.ContextualMultiSeasonalFTS(part + " " + ind, ix)

            model.train(sonda_treino, fs)

            Util.persist_obj(model, "models/sonda_cmsfts_" + part + "_" + str(max_part) + "_" + ind + ".pkl")



