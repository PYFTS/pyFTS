import pandas as pd
import matplotlib.pylab as plt
from pyFTS.data import TAIEX as tx
from pyFTS.common import Transformations


from pyFTS.data import SONDA
df = SONDA.get_dataframe()
train = df.iloc[0:578241] #three years
#test = df.iloc[1572480:2096640] #ears
del df

from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.common import Transformations, Util
from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

bc = Transformations.BoxCox(0)
tdiff = Transformations.Differential(1)

np = 10


model = mvfts.MVFTS("")

fig, axes = plt.subplots(nrows=5, ncols=1,figsize=[15,10])


sp = {'seasonality': DateTime.day_of_year , 'names': ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec']}

vmonth = variable.Variable("Month", data_label="datahora", partitioner=seasonal.TimeGridPartitioner, npart=12,
                           data=train, partitioner_specific=sp)
vmonth.partitioner.plot(axes[0])

sp = {'seasonality': DateTime.minute_of_day}

vhour = variable.Variable("Hour", data_label="datahora", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train, partitioner_specific=sp)

vhour.partitioner.plot(axes[1])

vavg = variable.Variable("Radiance", data_label="glo_avg", partitioner=Grid.GridPartitioner, npart=30,
                         data=train)

model1 = mvfts.MVFTS("")

model1.append_variable(vmonth)

model1.append_variable(vhour)

model1.append_variable(vavg)

model1.target_variable = vavg

#model1.fit(train, num_batches=60, save=True, batch_save=True, file_path='mvfts_sonda')


#model.fit(train, num_batches=60, save=True, batch_save=True, file_path='mvfts_sonda')

model1.fit(train, num_batches=200, save=True, batch_save=True, file_path='mvfts_sonda', distributed=True,
          nodes=['192.168.0.110'], batch_save_interval=10)


#model = Util.load_obj('mvfts_sonda')