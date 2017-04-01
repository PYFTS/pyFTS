from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import pandas as pd
import time
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cross_validation import KFold
from pyFTS.partitioners import partitioner, Grid, Huarng, Entropy, FCM
from pyFTS.benchmarks import Measures, naive, arima, ResidualAnalysis, ProbabilityDistribution
from pyFTS.common import Membership, FuzzySet, FLR, Transformations, Util
from pyFTS import fts, chen, yu, ismailefendi, sadaei, hofts, hwang,  pwfts, ifts
from pyFTS.benchmarks import  benchmarks

def get_first_order_models():
    return [chen.ConventionalFTS, yu.WeightedFTS, ismailefendi.ImprovedWeightedFTS,
                  sadaei.ExponentialyWeightedFTS]

def get_high_order_models():
    return [hofts.HighOrderFTS, pwfts.ProbabilisticWeightedFTS]


def run_first_order(method, partitioner, train_data, test_data, transformation = None, indexer=None ):
    mfts = method("")
    pttr = str(partitioner.__module__).split('.')[-1]
    _key = mfts.shortname + " " + pttr + " q = " + str(partitioner.partitions)
    mfts.partitioner = partitioner
    if transformation is not None:
        mfts.appendTransformation(transformation)

    try:
        _start = time.time()
        mfts.train(train_data, partitioner.sets)
        _end = time.time()
        times = _end - _start

        _start = time.time()
        _rmse, _smape, _u = benchmarks.get_point_statistics(test_data, mfts, indexer)
        _end = time.time()
        times += _end - _start
    except Exception as e:
        print(e)
        _rmse = np.nan
        _smape= np.nan
        _u = np.nan
        times = np.nan

    ret = {'key':_key, 'obj': mfts,  'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times }

    print(ret)

    return ret


def run_high_order(method, order, partitioner, train_data, test_data, transformation=None, indexer=None):
    mfts = method("")
    if order >= mfts.minOrder:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner
        if transformation is not None:
            mfts.appendTransformation(transformation)

        try:
            _start = time.time()
            mfts.train(train_data, partitioner.sets, order=order)
            _end = time.time()
            times = _end - _start

            _start = time.time()
            _rmse, _smape, _u = benchmarks.get_point_statistics(test_data, mfts, indexer)
            _end = time.time()
            times += _end - _start
        except Exception as e:
            print(e)
            _rmse = np.nan
            _smape = np.nan
            _u = np.nan
            times = np.nan

        ret = {'key': _key, 'obj': mfts, 'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times}

        print(ret)

        return ret

    return {'key': None, 'obj': mfts, 'rmse': np.nan, 'smape': np.nan, 'u': np.nan, 'time': np.nan}


def point_sliding_window(data, windowsize, train=0.8,models=None,partitioners=[Grid.GridPartitioner],
                   partitions=[10], max_order=3,transformation=None,indexer=None,dump=False,
                   save=False, file=None):

    num_cores = multiprocessing.cpu_count()

    objs = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    for ct, train,test in Util.sliding_window(data, windowsize, train):
        mocks = {}
        for partition in partitions:
            for partitioner in partitioners:
                pttr = str(partitioner.__module__).split('.')[-1]
                data_train_fs = partitioner(train, partition, transformation=transformation)

                results = Parallel(n_jobs=num_cores)(delayed(run_first_order)(m, deepcopy(data_train_fs), deepcopy(train), deepcopy(test), transformation)
                                                 for m in get_first_order_models())

                for tmp in results:
                    if tmp['key'] not in objs:
                        objs[tmp['key']] = tmp['obj']
                        rmse[tmp['key']] = []
                        smape[tmp['key']] = []
                        u[tmp['key']] = []
                        times[tmp['key']] = []
                    rmse[tmp['key']].append(tmp['rmse'])
                    smape[tmp['key']].append(tmp['smape'])
                    u[tmp['key']].append(tmp['u'])
                    times[tmp['key']].append(tmp['time'])

                for count, model in enumerate(get_high_order_models(), start=0):

                    results = Parallel(n_jobs=num_cores)(
                        delayed(run_high_order)(model, order, deepcopy(data_train_fs), deepcopy(train), deepcopy(test),
                                                 transformation)
                                                for order in np.arange(1, max_order + 1))

                    for tmp in results:
                        if tmp['key'] not in objs:
                            objs[tmp['key']] = tmp['obj']
                            rmse[tmp['key']] = []
                            smape[tmp['key']] = []
                            u[tmp['key']] = []
                            times[tmp['key']] = []
                        rmse[tmp['key']].append(tmp['rmse'])
                        smape[tmp['key']].append(tmp['smape'])
                        u[tmp['key']].append(tmp['u'])
                        times[tmp['key']].append(tmp['time'])
    ret = []
    for k in sorted(objs.keys()):
        try:
            mod = []
            tmp = objs[k]
            mod.append(tmp.shortname)
            mod.append(tmp.order )
            mod.append(tmp.partitioner.name)
            mod.append(tmp.partitioner.partitions)
            mod.append(np.round(np.nanmean(rmse[k]),2))
            mod.append(np.round(np.nanstd(rmse[k]), 2))
            mod.append(np.round(np.nanmean(smape[k]), 2))
            mod.append(np.round(np.nanstd(smape[k]), 2))
            mod.append(np.round(np.nanmean(u[k]), 2))
            mod.append(np.round(np.nanstd(u[k]), 2))
            mod.append(np.round(np.nanmean(times[k]), 4))
            mod.append(np.round(np.nanstd(times[k]), 4))
            mod.append(len(tmp))
            ret.append(mod)
        except Exception as ex:
            print("Erro ao salvar ",k)
            print("Exceção ", ex)

    columns = ["Model","Order","Scheme","Partitions","RMSEAVG","RMSESTD","SMAPEAVG","SMAPESTD","UAVG","USTD","TIMEAVG","TIMESTD","SIZE"]

    dat = pd.DataFrame(ret,columns=columns)

    if save: dat.to_csv(Util.uniquefilename(file),sep=";")

    return dat





