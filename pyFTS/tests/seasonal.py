import matplotlib.pylab as plt
from pyFTS.models.seasonal import partitioner, common
from pyFTS.partitioners import Util
from pyFTS.common import Membership


#fs = partitioner.TimeGridPartitioner(data=None, npart=12, seasonality=common.DateTime.day_of_year,
#                                     func=Membership.trapmf,
#                                     names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


#fs = partitioner.TimeGridPartitioner(None, 24, common.DateTime.minute_of_day, func=Membership.trapmf)

#fs = partitioner.TimeGridPartitioner(None, 7, common.DateTime.hour_of_week, func=Membership.trapmf)


#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 8])

#fs.plot(ax)
#plt.show()

"""
import pandas as pd

from pyFTS.data import SONDA
df = SONDA.get_dataframe()

df = df.drop(df[df.rain.values > 100].index)
df = df.drop(df[df.press.values < 800].index)
df = df.drop(df[df.humid.values < 15].index)

df.to_csv("SONDA_BSB_MOD.csv", sep=";", index=False)
"""

import os
print(os.getcwd())