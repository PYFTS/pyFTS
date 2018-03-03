"""
joblib Parallelized Benchmarks to FTS methods
"""

import datetime
import multiprocessing
import time
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed

from pyFTS.benchmarks import benchmarks, Util as bUtil
from pyFTS.common import Util
from pyFTS.partitioners import Grid


def run_point(mfts, partitioner, train_data, test_data, transformation=None, indexer=None):
    """
    Point forecast benchmark function to be executed on threads
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """
    pttr = str(partitioner.__module__).split('.')[-1]
    _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
    mfts.partitioner = partitioner
    if transformation is not None:
        mfts.append_transformation(transformation)

    try:
        _start = time.time()
        mfts.train(train_data, partitioner.sets, order=mfts.order)
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


def point_sliding_window(data, windowsize, train=0.8, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         save=False, file=None, sintetic=False):
    """
    Parallel sliding window benchmarks for FTS point forecasters
    :param data: 
    :param windowsize: size of sliding window
    :param train: percentual of sliding window data used to train the models
    :param models: FTS point forecasters
    :param partitioners: Universe of Discourse partitioner
    :param partitions: the max number of partitions on the Universe of Discourse 
    :param max_order: the max order of the models (for high order models)
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :param dump: 
    :param save: save results
    :param file: file path to save the results
    :param sintetic: if true only the average and standard deviation of the results
    :return: DataFrame with the results
    """
    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    num_cores = multiprocessing.cpu_count()

    pool = []

    objs = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    for model in benchmarks.get_point_methods():
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            pool.append(mfts)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train):
        experiments += 1

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                results = Parallel(n_jobs=num_cores)(
                    delayed(run_point)(deepcopy(m), deepcopy(data_train_fs), deepcopy(train), deepcopy(test),
                                       transformation)
                    for m in pool)

                for tmp in results:
                    if tmp['key'] not in objs:
                        objs[tmp['key']] = tmp['obj']
                        rmse[tmp['key']] = []
                        smape[tmp['key']] = []
                        u[tmp['key']] = []
                        times[tmp['key']] = []
                    rmse[tmp['key']].append_rhs(tmp['rmse'])
                    smape[tmp['key']].append_rhs(tmp['smape'])
                    u[tmp['key']].append_rhs(tmp['u'])
                    times[tmp['key']].append_rhs(tmp['time'])

    _process_end = time.time()

    print("Process End: {0: %H:%M:%S}".format(datetime.datetime.now()))

    print("Process Duration: {0}".format(_process_end - _process_start))

    return Util.save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u)


def run_interval(mfts, partitioner, train_data, test_data, transformation=None, indexer=None):
    """
    Interval forecast benchmark function to be executed on threads
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """
    pttr = str(partitioner.__module__).split('.')[-1]
    _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
    mfts.partitioner = partitioner
    if transformation is not None:
        mfts.append_transformation(transformation)

    try:
        _start = time.time()
        mfts.train(train_data, partitioner.sets, order=mfts.order)
        _end = time.time()
        times = _end - _start

        _start = time.time()
        _sharp, _res, _cov = benchmarks.get_interval_statistics(test_data, mfts)
        _end = time.time()
        times += _end - _start
    except Exception as e:
        print(e)
        _sharp = np.nan
        _res = np.nan
        _cov = np.nan
        times = np.nan

    ret = {'key': _key, 'obj': mfts, 'sharpness': _sharp, 'resolution': _res, 'coverage': _cov, 'time': times}

    print(ret)

    return ret


def interval_sliding_window(data, windowsize, train=0.8, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         save=False, file=None, sintetic=False):
    """
     Parallel sliding window benchmarks for FTS point_to_interval forecasters
     :param data: 
     :param windowsize: size of sliding window
     :param train: percentual of sliding window data used to train the models
     :param models: FTS point forecasters
     :param partitioners: Universe of Discourse partitioner
     :param partitions: the max number of partitions on the Universe of Discourse 
     :param max_order: the max order of the models (for high order models)
     :param transformation: data transformation
     :param indexer: seasonal indexer
     :param dump: 
     :param save: save results
     :param file: file path to save the results
     :param sintetic: if true only the average and standard deviation of the results
     :return: DataFrame with the results
     """
    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    num_cores = multiprocessing.cpu_count()

    pool = []

    objs = {}
    sharpness = {}
    resolution = {}
    coverage = {}
    times = {}

    for model in benchmarks.get_interval_methods():
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            pool.append(mfts)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train):
        experiments += 1

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                results = Parallel(n_jobs=num_cores)(
                    delayed(run_interval)(deepcopy(m), deepcopy(data_train_fs), deepcopy(train), deepcopy(test),
                                       transformation)
                    for m in pool)

                for tmp in results:
                    if tmp['key'] not in objs:
                        objs[tmp['key']] = tmp['obj']
                        sharpness[tmp['key']] = []
                        resolution[tmp['key']] = []
                        coverage[tmp['key']] = []
                        times[tmp['key']] = []

                    sharpness[tmp['key']].append_rhs(tmp['sharpness'])
                    resolution[tmp['key']].append_rhs(tmp['resolution'])
                    coverage[tmp['key']].append_rhs(tmp['coverage'])
                    times[tmp['key']].append_rhs(tmp['time'])

    _process_end = time.time()

    print("Process End: {0: %H:%M:%S}".format(datetime.datetime.now()))

    print("Process Duration: {0}".format(_process_end - _process_start))

    return Util.save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, sintetic, times)


def run_ahead(mfts, partitioner, train_data, test_data, steps, resolution, transformation=None, indexer=None):
    """
    Probabilistic m-step ahead forecast benchmark function to be executed on threads
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model 
    :param steps: 
    :param resolution: 
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """
    pttr = str(partitioner.__module__).split('.')[-1]
    _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
    mfts.partitioner = partitioner
    if transformation is not None:
        mfts.append_transformation(transformation)

    try:
        _start = time.time()
        mfts.train(train_data, partitioner.sets, order=mfts.order)
        _end = time.time()
        times = _end - _start

        _crps1, _crps2, _t1, _t2 = benchmarks.get_distribution_statistics(test_data, mfts, steps=steps,
                                                              resolution=resolution)
        _t1 += times
        _t2 += times
    except Exception as e:
        print(e)
        _crps1 = np.nan
        _crps2 = np.nan
        _t1 = np.nan
        _t2 = np.nan

    ret = {'key': _key, 'obj': mfts, 'CRPS_Interval': _crps1, 'CRPS_Distribution': _crps2, 'TIME_Interval': _t1, 'TIME_Distribution': _t2}

    print(ret)

    return ret


def ahead_sliding_window(data, windowsize, train, steps,resolution, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         save=False, file=None, sintetic=False):
    """
    Parallel sliding window benchmarks for FTS probabilistic forecasters
    :param data: 
    :param windowsize: size of sliding window
    :param train: percentual of sliding window data used to train the models
    :param steps: 
    :param resolution: 
    :param models: FTS point forecasters
    :param partitioners: Universe of Discourse partitioner
    :param partitions: the max number of partitions on the Universe of Discourse 
    :param max_order: the max order of the models (for high order models)
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :param dump: 
    :param save: save results
    :param file: file path to save the results
    :param sintetic: if true only the average and standard deviation of the results
    :return: DataFrame with the results 
    """
    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    num_cores = multiprocessing.cpu_count()

    pool = []

    objs = {}
    crps_interval = {}
    crps_distr = {}
    times1 = {}
    times2 = {}

    for model in benchmarks.get_interval_methods():
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            pool.append(mfts)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train):
        experiments += 1

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                results = Parallel(n_jobs=num_cores)(
                    delayed(run_ahead)(deepcopy(m), deepcopy(data_train_fs), deepcopy(train), deepcopy(test),
                                       steps, resolution, transformation)
                    for m in pool)

                for tmp in results:
                    if tmp['key'] not in objs:
                        objs[tmp['key']] = tmp['obj']
                        crps_interval[tmp['key']] = []
                        crps_distr[tmp['key']] = []
                        times1[tmp['key']] = []
                        times2[tmp['key']] = []

                    crps_interval[tmp['key']].append_rhs(tmp['CRPS_Interval'])
                    crps_distr[tmp['key']].append_rhs(tmp['CRPS_Distribution'])
                    times1[tmp['key']].append_rhs(tmp['TIME_Interval'])
                    times2[tmp['key']].append_rhs(tmp['TIME_Distribution'])

    _process_end = time.time()

    print("Process End: {0: %H:%M:%S}".format(datetime.datetime.now()))

    print("Process Duration: {0}".format(_process_end - _process_start))

    return Util.save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, sintetic)
