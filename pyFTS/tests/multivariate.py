import pandas as pd
import matplotlib.pylab as plt
from pyFTS.data import TAIEX as tx
from pyFTS.common import Transformations


bc = Transformations.BoxCox(0)
diff = Transformations.Differential(1)

df = tx.get_dataframe()
df = df.dropna()
#df.loc[2209]
train = df.iloc[2000:2500]
test = df.iloc[2500:3000]

from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.models.multivariate import common, variable

model = common.MVFTS("")

#fig, axes = plt.subplots(nrows=5, ncols=1,figsize=[10,10])

vopen = variable.Variable("Open", data_label="Openly", partitioner=Grid.GridPartitioner, npart=40, data=df)
model.append_variable(vopen)
#vopen.partitioner.plot(axes[0])
vhigh = variable.Variable("High", data_label="Highest", partitioner=Grid.GridPartitioner, npart=40, data=df)#train)
model.append_variable(vhigh)
#vhigh.partitioner.plot(axes[1])
vlow = variable.Variable("Low", data_label="Lowermost", partitioner=Grid.GridPartitioner, npart=40, data=df)#train)
model.append_variable(vlow)
#vlow.partitioner.plot(axes[2])
vclose = variable.Variable("Close", data_label="Close", partitioner=Grid.GridPartitioner, npart=40, data=df)#train)
model.append_variable(vclose)
#vclose.partitioner.plot(axes[3])
vvol = variable.Variable("Volume", data_label="Volume", partitioner=Grid.GridPartitioner, npart=100, data=df,
                         transformation=bc)#train)
model.append_variable(vvol)
#vvol.partitioner.plot(axes[4])

model.target_variable = vvol

#plt.tight_layout()
model.train(train)

forecasted = model.forecast(test)

print([round(k,0) for k in test['Volume'].values.tolist()])
print([round(k,0) for k in forecasted])