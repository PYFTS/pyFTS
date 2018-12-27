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



def get_partitioner(shared_partitioner):
    """

    :param part:
    :return:
    """
    fs_tmp = Simple.SimplePartitioner()

    for fset in shared_partitioner.value.keys():
        fz = shared_partitioner.value[fset]
        fs_tmp.append(fset, fz.mf, fz.parameters)

    return fs_tmp


def slave_train(data, shared_method, shared_partitioner, shared_order):
    """

    :param data:
    :return:
    """

    model = shared_method.value(partitioner=get_partitioner(shared_partitioner),
                                order=shared_order.value)

    ndata = [k for k in data]

    model.train(ndata)

    return [(k, model.flrgs[k]) for k in model.flrgs]


def distributed_train(model, data, url='spark://192.168.0.110:7077', app='pyFTS'):
    """


    :param model:
    :param data:
    :param url:
    :param app:
    :return:
    """

    conf = SparkConf()
    conf.setMaster(url)
    conf.setAppName(app)

    with SparkContext(conf=conf) as context:
        shared_partitioner = context.broadcast(model.partitioner.sets)
        shared_order = context.broadcast(model.order)
        shared_method = context.broadcast(type(model))

        func = lambda x: slave_train(x, shared_method, shared_partitioner, shared_order)

        flrgs = context.parallelize(data).mapPartitions(func)

        for k in flrgs.collect():
            model.append_rule(k[1])

        return model



def distributed_predict(data, model, url='spark://192.168.0.110:7077', app='pyFTS'):
    return None
