import pandas as pd
import matplotlib.pylab as plt
from pyFTS.data import TAIEX, Malaysia
from pyFTS.common import Transformations

from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.common import Transformations, Util
from pyFTS.models import pwfts
from pyFTS.models.multivariate import common, variable, mvfts, wmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

bc = Transformations.BoxCox(0)
tdiff = Transformations.Differential(1)

from pyFTS.models.multivariate import common, variable, mvfts, cmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

from pyFTS.data import henon
df = henon.get_dataframe(iterations=1000)

from pyFTS.models.multivariate import variable, cmvfts

vx = variable.Variable("x", data_label="x", partitioner=Grid.GridPartitioner, npart=15, data=df)
vy = variable.Variable("y", data_label="y", partitioner=Grid.GridPartitioner, npart=15, data=df)

model = cmvfts.ClusteredMVFTS(pre_fuzzyfy=False, knn=3, order=2, fts_method=pwfts.ProbabilisticWeightedFTS)
model.append_variable(vx)
model.append_variable(vy)
model.target_variable = vx

model.fit(df.iloc[:800])

test = df.iloc[800:]

forecasts = model.predict(test, type='multivariate')

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[15,7])
ax[0][0].plot(test['x'].values)
ax[0][0].plot(forecasts['x'].values)
ax[0][1].scatter(test['x'].values,test['y'].values)
ax[0][1].scatter(forecasts['x'].values,forecasts['y'].values)
ax[1][0].scatter(test['y'].values,test['x'].values)
ax[1][0].scatter(forecasts['y'].values,forecasts['x'].values)
ax[1][1].plot(test['y'].values)
ax[1][1].plot(forecasts['y'].values)