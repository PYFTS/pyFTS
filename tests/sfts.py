#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.partitioners import Grid
from pyFTS.common import FLR,FuzzySet,Membership,Transformations
from pyFTS import fts,sfts
from pyFTS.models import msfts
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import Measures

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

sonda = pd.read_csv("DataSets/SONDA_BSB_CLEAN.csv", sep=";")

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

ix = SeasonalIndexer.DataFrameSeasonalIndexer(['day','min'],[30, 60],'glo_avg')

fs = Grid.GridPartitionerTrimf(ix.get_data(sonda_treino),20)

#mfts = msfts.MultiSeasonalFTS("",ix)

#mfts.train(sonda_teste,fs)

#print(str(mfts))

#[10, 508]

flrs = FLR.generateIndexedFLRs(fs, ix, sonda_treino[110000:111450])

for i in flrs:  #ix.get_data(sonda_treino[111430:111450]):
    print(i)