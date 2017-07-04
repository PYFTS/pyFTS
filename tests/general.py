#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.partitioners import Grid, Entropy, FCM, Huarng
from pyFTS.common import FLR,FuzzySet,Membership,Transformations, Util as cUtil
from pyFTS import fts,hofts,ifts,pwfts,tree, chen
#from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import naive, arima
from pyFTS.benchmarks import Measures
from numpy import random
from pyFTS.models.seasonal import SeasonalIndexer

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

diff = Transformations.Differential(1)
#ix = SeasonalIndexer.LinearSeasonalIndexer([12, 24], [720, 1],[False, False])

"""
DATASETS
"""

#enrollments = pd.read_csv("DataSets/Enrollments.csv", sep=";")
#enrollments = np.array(enrollments["Enrollments"])

#passengers = pd.read_csv("DataSets/AirPassengers.csv", sep=",")
#passengers = np.array(passengers["Passengers"])

#sunspots = pd.read_csv("DataSets/sunspots.csv", sep=",")
#sunspots = np.array(sunspots["SUNACTIVITY"])

#gauss = random.normal(0,1.0,5000)
#gauss_teste = random.normal(0,1.0,400)

#taiexpd = pd.read_csv("DataSets/TAIEX.csv", sep=",")
#taiex = np.array(taiexpd["avg"][:5000])
#del(taiexpd)

#nasdaqpd = pd.read_csv("DataSets/NASDAQ_IXIC.csv", sep=",")
#nasdaq = np.array(nasdaqpd["avg"][0:5000])
#del(nasdaqpd)

#sp500pd = pd.read_csv("DataSets/S&P500.csv", sep=",")
#sp500 = np.array(sp500pd["Avg"][11000:])
#del(sp500pd)

#sondapd = pd.read_csv("DataSets/SONDA_BSB_HOURLY_AVG.csv", sep=";")
#sondapd = sondapd.dropna(axis=0, how='any')
#sonda = np.array(sondapd["glo_avg"])
#del(sondapd)

#bestpd = pd.read_csv("DataSets/BEST_TAVG.csv", sep=";")
#best = np.array(bestpd["Anomaly"])
#del(bestpd)

#print(lag)
#print(a)

sonda = pd.read_csv("DataSets/SONDA_BSB_MOD.csv", sep=";")

sonda['data'] = pd.to_datetime(sonda['data'])

sonda = sonda[:][527041:]

sonda.index = np.arange(0,len(sonda.index))

sonda_treino = sonda[:1051200]
sonda_teste = sonda[1051901:1051910]

ix_m15 = SeasonalIndexer.DateTimeSeasonalIndexer('data',[SeasonalIndexer.DateTime.minute],[15],'glo_avg', name='m15')

fs1 = Grid.GridPartitioner(sonda_treino,50,transformation=diff, indexer=ix_m15)


'''
from pyFTS.models.seasonal import SeasonalIndexer

indexers = []

for i in ["models/sonda_ix_Mhm15.pkl"]: #, "models/sonda_ix_m15.pkl", "models/sonda_ix_Mh.pkl", ]:
    obj = cUtil.load_obj(i)
    indexers.append( obj )
    print(obj)

partitioners = []

transformations = [""] #, "_diff"]
for max_part in [30, 40, 50, 60, 70, 80, 90]:
    for t in transformations:
        obj = cUtil.load_obj("models/sonda_fs_grid_" + str(max_part) + t + ".pkl")
        partitioners.append( obj )
        print(obj)


from pyFTS.ensemble import ensemble, multiseasonal

fts = multiseasonal.SeasonalEnsembleFTS("sonda_msfts_Mhm15")

fts.indexers = indexers
fts.partitioners = partitioners

fts.indexer = indexers[0]

fts.train(sonda_treino, sets=None)
'''
#'''

#ix = cUtil.load_obj("models/sonda_ix_m15.pkl")

#ftse = cUtil.load_obj("models/msfts_Grid40_diff_Mhm15.pkl")

#ftse.indexer = ix

#ftse.update_uod(sonda_treino)

#tmp = ftse.forecastDistribution(sonda_teste,h=1)

#tmp = ftse.forecast(sonda_teste,h=1)

#tmp[5].plot()
#'''

'''
from pyFTS.benchmarks import benchmarks as bchmk
#from pyFTS.benchmarks import distributed_benchmarks as bchmk
#from pyFTS.benchmarks import parallel_benchmarks as bchmk
from pyFTS.benchmarks import Util
from pyFTS.benchmarks import arima, quantreg, Measures

#Util.cast_dataframe_to_synthetic_point("experiments/taiex_point_analitic.csv","experiments/taiex_point_sintetic.csv",11)

#Util.plot_dataframe_point("experiments/taiex_point_sintetic.csv","experiments/taiex_point_analitic.csv",11)
"""
arima100 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima100.train(passengers, None, order=(1,0,0))

arima101 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima101.train(passengers, None, order=(1,0,1))

arima200 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima200.train(passengers, None, order=(2,0,0))

arima201 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima201.train(passengers, None, order=(2,0,1))


#tmp = quantreg.QuantileRegression("", alpha=0.25, dist=True)
#tmp.appendTransformation(diff)
#tmp.train(sunspots[:150], None, order=1)
#teste = tmp.forecastAheadInterval(sunspots[150:155], 5)
#teste = tmp.forecastAheadDistribution(nasdaq[1600:1604], steps=5, resolution=50)

bchmk.plot_compared_series(enrollments,[tmp], ['blue','red'], points=False, intervals=True)

#print(sunspots[150:155])
#print(teste)

#kk = Measures.get_interval_statistics(nasdaq[1600:1605], tmp)

#print(kk)
"""


"""
bchmk.point_sliding_window(sonda, 9000, train=0.8, inc=0.4,#models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(10,200,step=10), #transformation=diff,
                     dump=True, save=True, file="experiments/sondaws_point_analytic.csv",
                     nodes=['192.168.0.103', '192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])



bchmk.point_sliding_window(sonda, 9000, train=0.8, inc=0.4, #models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(3,20,step=2), #transformation=diff,
                     dump=True, save=True, file="experiments/sondaws_point_analytic_diff.csv",
                     nodes=['192.168.0.103', '192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])





bchmk.interval_sliding_window(best, 5000, train=0.8, inc=0.8,#models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(10,200,step=10),
                     dump=True, save=True, file="experiments/best"
                                                "_interval_analytic.csv",
                     nodes=['192.168.0.103', '192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])



bchmk.interval_sliding_window(taiex, 2000, train=0.8, inc=0.1, #models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(3,20,step=2), transformation=diff,
                     dump=True, save=True, file="experiments/taiex_interval_analytic_diff.csv",
                     nodes=['192.168.0.103', '192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])





bchmk.ahead_sliding_window(sonda, 10000, steps=10, resolution=10, train=0.2, inc=0.2,
                     partitioners=[Grid.GridPartitioner],
                     partitions= np.arange(10,200,step=10), indexer=ix,
                     dump=True, save=True, file="experiments/sondawind_ahead_analytic.csv",
                     nodes=['192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])


bchmk.ahead_sliding_window(sonda, 10000, steps=10, resolution=10, train=0.2, inc=0.2,
                     partitioners=[Grid.GridPartitioner],
                     partitions= np.arange(3,20,step=2), transformation=diff, indexer=ix,
                     dump=True, save=True, file="experiments/sondawind_ahead_analytic_diff.csv",
                     nodes=['192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])

"""

from pyFTS import pwfts
from pyFTS.common import Transformations
from pyFTS.partitioners import Grid

#diff = Transformations.Differential(1)
#fs = Grid.GridPartitioner(best, 190) #, transformation=diff)


#model = pwfts.ProbabilisticWeightedFTS("FTS 1")
#model.appendTransformation(diff)
#model.train(best[0:1600],fs.sets, order=3)

#bchmk.plot_compared_intervals_ahead(best[1600:1700],[model], ['blue','red'],
#                                    distributions=[True], save=True, file="pictures/best_ahead_forecasts",
#                                    time_from=40, time_to=60, resolution=100)
'''
experiments = [
    ["experiments/taiex_point_synthetic_diff.csv","experiments/taiex_point_analytic_diff.csv",16],
    ["experiments/nasdaq_point_synthetic_diff.csv","experiments/nasdaq_point_analytic_diff.csv", 11],
    ["experiments/sp500_point_synthetic_diff.csv","experiments/sp500_point_analytic_diff.csv", 21],
    ["experiments/best_point_synthetic_diff.csv","experiments/best_point_analytic_diff.csv", 13],
    ["experiments/sondasun_point_synthetic_diff.csv","experiments/sondasun_point_analytic_diff.csv", 15],
    ["experiments/sondawind_point_synthetic_diff.csv","experiments/sondawind_point_analytic_diff.csv", 8],
    ["experiments/gauss_point_synthetic_diff.csv","experiments/gauss_point_analytic_diff.csv", 16]
]

Util.unified_scaled_point(experiments,tam=[15,8],save=True,file="pictures/unified_experiments_point.png",
                         ignore=['ARIMA(1,0,0)','ARIMA(2,0,0)','ARIMA(2,0,1)','ARIMA(2,0,2)','QAR(2)'],
                         replace=[['ARIMA','ARIMA'],['QAR','QAR']])
'''

'''
experiments = [
    ["experiments/taiex_interval_synthetic.csv","experiments/taiex_interval_analytic.csv",16],
    ["experiments/nasdaq_interval_synthetic_diff.csv","experiments/nasdaq_interval_analytic_diff.csv",11],
    ["experiments/sp500_interval_synthetic_diff.csv","experiments/sp500_interval_analytic_diff.csv", 11],
    ["experiments/best_interval_synthetic_diff.csv","experiments/best_interval_analytic_diff.csv",13],
    ["experiments/sondasun_interval_synthetic_diff.csv","experiments/sondasun_interval_analytic_diff.csv",8],
    ["experiments/sondawind_interval_synthetic_diff.csv","experiments/sondawind_interval_analytic_diff.csv",8],
    ["experiments/gauss_interval_synthetic_diff.csv","experiments/gauss_interval_analytic_diff.csv", 8]
]

Util.unified_scaled_interval(experiments,tam=[15,8],save=True,file="pictures/unified_experiments_interval.png",
                             ignore=['ARIMA(1,0,0)', 'ARIMA(2,0,0)', 'ARIMA(2,0,1)', 'ARIMA(2,0,2)', 'QAR(2)'],
                             replace=[['ARIMA(1,0,1) - 0.05', 'ARIMA 0.05'], ['ARIMA(1,0,1) - 0.25', 'ARIMA 0.25'],
                                      ['QAR(1) - 0.05', 'QAR 0.05'], ['QAR(1) - 0.25', 'QAR 0.25']])

Util.unified_scaled_interval_pinball(experiments,tam=[15,8],save=True,file="pictures/unified_experiments_interval_pinball.png",
                             ignore=['ARIMA(1,0,0)', 'ARIMA(2,0,0)', 'ARIMA(2,0,1)', 'ARIMA(2,0,2)', 'QAR(2)'],
                             replace=[['ARIMA(1,0,1) - 0.05', 'ARIMA 0.05'], ['ARIMA(1,0,1) - 0.25', 'ARIMA 0.25'],
                                      ['QAR(1) - 0.05', 'QAR 0.05'], ['QAR(1) - 0.25', 'QAR 0.25']])

'''

'''
experiments = [
    ["experiments/taiex_ahead_synthetic_diff.csv","experiments/taiex_ahead_analytic_diff.csv",16],
    ["experiments/nasdaq_ahead_synthetic_diff.csv","experiments/nasdaq_ahead_analytic_diff.csv",11],
    ["experiments/sp500_ahead_synthetic_diff.csv","experiments/sp500_ahead_analytic_diff.csv", 21],
    ["experiments/best_ahead_synthetic_diff.csv","experiments/best_ahead_analytic_diff.csv", 24],
    ["experiments/sondasun_ahead_synthetic_diff.csv","experiments/sondasun_ahead_analytic_diff.csv",13],
    ["experiments/sondawind_ahead_synthetic_diff.csv","experiments/sondawind_ahead_analytic_diff.csv", 13],
    ["experiments/gauss_ahead_synthetic_diff.csv","experiments/gauss_ahead_analytic_diff.csv",16]
]

Util.unified_scaled_ahead(experiments,tam=[15,8],save=True,file="pictures/unified_experiments_ahead.png",
                          ignore=['ARIMA(1,0,0)', 'ARIMA(0,0,1)', 'ARIMA(2,0,0)', 'ARIMA(2,0,1)',
                                  'ARIMA(2,0,2)', 'QAR(2)', 'ARIMA0.05'],
                          replace=[['ARIMA(1,0,1) - 0.05', 'ARIMA 0.05'], ['ARIMA(1,0,1) - 0.25', 'ARIMA 0.25'],
                                   ['QAR(1) - 0.05', 'QAR 0.05'], ['QAR(1) - 0.25', 'QAR 0.25']])



'''

'''
from pyFTS.partitioners import Grid

from pyFTS import sfts



#print(ix.get_season_of_data(best[:2000]))

#print(ix.get_season_by_index(45))

#ix = SeasonalIndexer.LinearSeasonalIndexer([720,24],[False,True,False])

#print(ix.get_season_of_data(sonda[6500:9000])[-20:])

diff = Transformations.Differential(1)

fs = Grid.GridPartitioner(sonda[:9000], 10, transformation=diff)


'''
tmp = sfts.SeasonalFTS("")
tmp.indexer = ix
tmp.appendTransformation(diff)

#tmp = pwfts.ProbabilisticWeightedFTS("")

#tmp.appendTransformation(diff)

tmp.train(sonda[:9000], fs.sets, order=1)

x = tmp.forecast(sonda[:1610])

#print(taiex[1600:1610])
print(x)
'''