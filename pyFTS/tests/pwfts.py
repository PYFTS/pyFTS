import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import importlib
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pyFTS.common import Util
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import pwfts,hofts,ifts
from pyFTS.models.multivariate import granular, grid
from pyFTS.partitioners import Grid, Util as pUtil

from pyFTS.models.multivariate import common, variable, mvfts, wmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.common import Membership

def sample_by_hour(data):
    return [np.nanmean(data[k:k+60]) for k in np.arange(0,len(data),60)]

def sample_date_by_hour(data):
    return [data[k] for k in np.arange(0,len(data),60)]

from pyFTS.data import SONDA

sonda = SONDA.get_dataframe()[['datahora','glo_avg','ws_10m']]

sonda = sonda.drop(sonda.index[np.where(sonda["ws_10m"] <= 0.01)])
sonda = sonda.drop(sonda.index[np.where(sonda["glo_avg"] <= 0.01)])
sonda = sonda.dropna()
sonda['datahora'] = pd.to_datetime(sonda["datahora"], format='%Y-%m-%d %H:%M:%S')


var = {
    'datahora': sample_date_by_hour(sonda['datahora'].values),
    'glo_avg': sample_by_hour(sonda['glo_avg'].values),
    'ws_10m': sample_by_hour(sonda['ws_10m'].values),
}

df = pd.DataFrame(var)

train_mv = df.iloc[:9000]
test_mv = df.iloc[9000:10000]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[10,3])

sp = {'seasonality': DateTime.month, 'names': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}

vmonth = variable.Variable("Month", data_label="datahora", partitioner=seasonal.TimeGridPartitioner, npart=12,
                          data=train_mv, partitioner_specific=sp, alpha_cut=.3)

vmonth.partitioner.plot(ax[0])

vwin = variable.Variable("Wind", data_label="ws_10m", alias='wind',
                         partitioner=Grid.GridPartitioner, npart=15, func=Membership.gaussmf,
                         data=train_mv, alpha_cut=.25)

vwin.partitioner.plot(ax[1])

plt.tight_layout()

order = 3
knn = 2

model = granular.GranularWMVFTS(explanatory_variables=[vmonth, vwin], target_variable=vwin,
                                fts_method=pwfts.ProbabilisticWeightedFTS, fuzzyfy_mode='both',
                                order=order, knn=knn)

model.fit(train_mv)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,3])
ax.plot(test_mv['ws_10m'].values[:100], label='original')

forecasts = model.predict(test_mv.iloc[:100], type='distribution')

Util.plot_distribution2(forecasts, test_mv['ws_10m'].values[:100], start_at=model.order-1, ax=ax)

