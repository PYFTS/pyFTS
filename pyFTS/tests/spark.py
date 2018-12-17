import numpy as np
import pandas as pd

from pyFTS.data import Enrollments, TAIEX
from pyFTS.partitioners import Grid, Simple
from pyFTS.models import hofts

from pyspark import SparkConf
from pyspark import SparkContext

import os
# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'

conf = SparkConf()
conf.setMaster('spark://192.168.0.110:7077')
conf.setAppName('pyFTS')

data = TAIEX.get_data()

fs = Grid.GridPartitioner(data=data, npart=50)


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





