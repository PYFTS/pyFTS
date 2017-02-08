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
from pyFTS import fts,sfts
from pyFTS.models import msfts
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import Measures

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

sonda = pd.read_csv("DataSets/SONDA_BSB_MOD.csv", sep=";")

sonda['data'] = pd.to_datetime(sonda['data'])

sonda = sonda[:][527041:]

sonda.index = np.arange(0,len(sonda.index))

#data = []

#for i in sonda.index:

    #inst = []

    #year = int( sonda["year"][i] )
    #day_of_year = int( sonda["day"][i] )
    #minute = int (sonda["min"][i] )

    #glo_avg = sonda["glo_avg"][i]

    #inst.append(  datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1, minutes=minute) )

    #inst.append( glo_avg )

    #data.append(inst)

#nov = pd.DataFrame(data,columns=["data","glo_avg"])

#nov.to_csv("DataSets/SONDA_BSB_MOD.csv", sep=";")

sonda_treino = sonda[:1051200]
sonda_teste = sonda[1051201:]


#res = bchmk.simpleSearch_RMSE(sonda_treino, sonda_teste,
#                              sfts.SeasonalFTS,np.arange(3,30),[1],parameters=1440,
#                              tam=[15,8], plotforecasts=False,elev=45, azim=40,
#                               save=False,file="pictures/sonda_sfts_error_surface", intervals=False)

from pyFTS.models.seasonal import SeasonalIndexer
from pyFTS.models import msfts
from pyFTS.common import FLR

ix = SeasonalIndexer.DateTimeSeasonalIndexer('data',[SeasonalIndexer.DateTime.month,
                                                     SeasonalIndexer.DateTime.hour, SeasonalIndexer.DateTime.minute],
                                             [None, None,15],'glo_avg')

tmp = ix.get_data(sonda_treino)
for max_part in [10, 20, 30, 40, 50]:

    fs1 = Grid.GridPartitionerTrimf(tmp,max_part)

    Util.persist_obj(fs1,"models/sonda_fs_grid_" + str(max_part) + ".pkl")

    fs2 = FCM.FCMPartitionerTrimf(tmp, max_part)

    Util.persist_obj(fs2, "models/sonda_fs_fcm_" + str(max_part) + ".pkl")

    fs3 = Entropy.EntropyPartitionerTrimf(tmp, max_part)

    Util.persist_obj(fs3, "models/sonda_fs_entropy_" + str(max_part) + ".pkl")


#fs = Util.load_obj("models/sonda_fs_grid_50.pkl")

#for f in fs:
#    print(f)

#mfts = msfts.MultiSeasonalFTS("",ix)

#mfts.train(sonda_treino,fs)

#print(str(mfts))

#plt.plot(mfts.forecast(sonda_teste))

#[10, 508]

#flrs = FLR.generateIndexedFLRs(fs, ix, sonda_treino[110000:111450])

#for i in mfts.forecast(sonda_teste):
#    print(i)