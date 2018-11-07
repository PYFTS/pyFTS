import pandas as pd
import matplotlib.pylab as plt
from pyFTS.data import TAIEX as tx
from pyFTS.common import Transformations


from pyFTS.data import Malaysia

dataset = Malaysia.get_dataframe()

dataset["date"] = pd.to_datetime(dataset["time"], format='%m/%d/%y %I:%M %p')

from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.common import Transformations, Util
from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

bc = Transformations.BoxCox(0)
tdiff = Transformations.Differential(1)

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

mv_train = dataset.iloc[:100000]

sp = {'seasonality': DateTime.month , 'names': ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec']}

vmonth = variable.Variable("Month", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=12, 
                         data=mv_train, partitioner_specific=sp)

sp = {'seasonality': DateTime.day_of_week, 'names': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}

vday = variable.Variable("Weekday", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=7, 
                        data=mv_train, partitioner_specific=sp)

sp = {'seasonality': DateTime.hour_of_day}

vhour = variable.Variable("Hour", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=24, 
                        data=mv_train, partitioner_specific=sp)

vload = variable.Variable("load", data_label="load", partitioner=Grid.GridPartitioner, npart=10, 
                       data=mv_train)

vtemperature = variable.Variable("temperature", data_label="temperature", partitioner=Grid.GridPartitioner, npart=10, 
                       data=mv_train)


variables = {
    'month': vmonth,
    'day': vday,
    'hour': vhour,
    'temperature': vtemperature,
    'load': vload
}

var_list = [k for k in variables.keys()]

models = []

import itertools

for k in [itertools.combinations(var_list, r) for r in range(2,len(var_list))]:
    for x in k:
      model = mvfts.MVFTS()
      print(x)
      for w in x:
        model.append_variable(variables[w])
        model.shortname += ' ' + w
      model.target_variable = vload
      model.fit(mv_train)
      models.append(model)
