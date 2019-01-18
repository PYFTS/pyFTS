import numpy as np
import pandas as pd
import time

from pyFTS.data import Enrollments, TAIEX, SONDA
from pyFTS.partitioners import Grid, Simple
from pyFTS.models import hofts

from pyspark import SparkConf
from pyspark import SparkContext

import os
# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
#'''
data = SONDA.get_data('glo_avg')

fs = Grid.GridPartitioner(data=data, npart=50)

model = hofts.WeightedHighOrderFTS(partitioner=fs, order=2)

_s1 = time.time()
model.fit(data, distributed='spark', url='spark://192.168.0.106:7077')
_s2 = time.time()

print(_s2-_s1)

#model.fit(data, distributed='dispy', nodes=['192.168.0.110'])
'''

from pyFTS.models.multivariate import common, variable, mvfts, wmvfts, cmvfts, grid
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

dataset = pd.read_csv('/home/petronio/Downloads/kalang.csv', sep=',')

dataset['date'] = pd.to_datetime(dataset["date"], format='%Y-%m-%d %H:%M:%S')

train_mv = dataset.iloc[:24505]
test_mv = dataset.iloc[24505:]

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp, data_type=pd.datetime, mask='%Y-%m-%d %H:%M:%S')

vvalue = variable.Variable("Pollution", data_label="value", alias='value',
                         partitioner=Grid.GridPartitioner, npart=35, data_type=np.float64,
                         data=train_mv)

fs = grid.GridCluster(explanatory_variables=[vhour, vvalue], target_variable=vvalue)
#model = wmvfts.WeightedMVFTS(explanatory_variables=[vhour, vvalue], target_variable=vvalue)
model = cmvfts.ClusteredMVFTS(explanatory_variables=[vhour, vvalue], target_variable=vvalue,
                              partitioner=fs)

model.fit(train_mv, distributed='spark', url='spark://192.168.0.106:7077')
#'''
print(model)

'''
def fun(x):
    return (x, x % 2)


def get_fs():
    fs_tmp = Simple.SimplePartitioner()

    for fset in part.value.keys():
        fz = part.value[fset]
        fs_tmp.append(fset, fz.mf, fz.parameters)

    return fs_tmp

def fuzzyfy(x):

    fs_tmp = get_fs()

    ret = []

    for k in x:
        ret.append(fs_tmp.fuzzyfy(k, mode='both'))

    return ret


def train(fuzzyfied):
    model = hofts.WeightedHighOrderFTS(partitioner=get_fs(), order=order.value)

    ndata = [k for k in fuzzyfied]

    model.train(ndata)

    return [(k, model.flrgs[k]) for k in model.flrgs]


with SparkContext(conf=conf) as sc:

    part = sc.broadcast(fs.sets)

    order = sc.broadcast(2)

    #ret = sc.parallelize(np.arange(0,100)).map(fun)

    #fuzzyfied = sc.parallelize(data).mapPartitions(fuzzyfy)

    flrgs = sc.parallelize(data).mapPartitions(train)

    model = hofts.WeightedHighOrderFTS(partitioner=fs, order=order.value)

    for k in flrgs.collect():
        model.append_rule(k[1])

    print(model)

'''



