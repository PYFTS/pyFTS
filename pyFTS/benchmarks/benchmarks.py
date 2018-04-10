#!/usr/bin/python
# -*- coding: utf8 -*-

"""Benchmarks to FTS methods"""


import datetime
import time
from copy import deepcopy

import matplotlib as plt
import matplotlib.cm as cmx
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pyFTS.probabilistic import ProbabilityDistribution
from pyFTS.models import song, chen, yu, ismailefendi, sadaei, hofts, pwfts, ifts, cheng, hwang
from pyFTS.models.ensemble import ensemble
from pyFTS.benchmarks import Measures, naive, arima, ResidualAnalysis, quantreg
from pyFTS.benchmarks import Util as bUtil
from pyFTS.common import Util as cUtil
# from sklearn.cross_validation import KFold
from pyFTS.partitioners import Grid
from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

colors = ['grey', 'darkgrey', 'rosybrown', 'maroon', 'red','orange', 'gold', 'yellow', 'olive', 'green',
          'darkgreen', 'cyan', 'lightblue','blue', 'darkblue', 'purple', 'darkviolet' ]

ncol = len(colors)

styles = ['-','--','-.',':','.']

nsty = len(styles)


def __pop(key, default, kwargs):
    if key in kwargs:
        return kwargs.pop(key)
    else:
        return default


def sliding_window_benchmarks(data, windowsize, train=0.8, **kwargs):
    """
    Sliding window benchmarks for FTS point forecasters
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
    :param benchmark_methods: Non FTS models to benchmark
    :param benchmark_methods_parameters: Non FTS models parameters
    :param save: save results
    :param file: file path to save the results
    :param sintetic: if true only the average and standard deviation of the results
    :return: DataFrame with the results
    """
    distributed = __pop('distributed', False, kwargs)
    save = __pop('save', False, kwargs)

    transformation = kwargs.get('transformation', None)
    progress = kwargs.get('progress', None)
    type = kwargs.get("type", 'point')

    orders = __pop("orders", [1,2,3], kwargs)

    partitioners_models = __pop("partitioners_models", None, kwargs)
    partitioners_methods = __pop("partitioners_methods", [Grid.GridPartitioner], kwargs)
    partitions = __pop("partitions", [10], kwargs)

    methods = __pop('methods', None, kwargs)

    models = __pop('models', None, kwargs)

    pool = [] if models is None else models

    if models is None and methods is None:
        if type  == 'point':
            methods = get_point_methods()
        elif type == 'interval':
            methods = get_interval_methods()
        elif type == 'distribution':
            methods = get_probabilistic_methods()

    if models is None:
        for method in methods:
            mfts = method("")

            if mfts.is_high_order:
                for order in orders:
                    if order >= mfts.min_order:
                        mfts = method("")
                        mfts.order = order
                        pool.append(mfts)
            else:
                mfts.order = 1
                pool.append(mfts)

    benchmark_models = __pop("benchmark_models", None, kwargs)

    benchmark_methods = __pop("benchmark_methods", None, kwargs)
    benchmark_methods_parameters = __pop("benchmark_methods_parameters", None, kwargs)

    if benchmark_models != False:

        if benchmark_models is None and benchmark_methods is None:
            if type == 'point'or type  == 'partition':
                benchmark_methods = get_benchmark_point_methods()
            elif type == 'interval':
                benchmark_methods = get_benchmark_interval_methods()
            elif type == 'distribution':
                benchmark_methods = get_benchmark_probabilistic_methods()

        if benchmark_models is not None:
            pool.extend(benchmark_models)
        elif benchmark_methods is not None:
            for count, model in enumerate(benchmark_methods, start=0):
                par = benchmark_methods_parameters[count]
                mfts = model(str(par if par is not None else ""))
                mfts.order = par
                pool.append(mfts)

    if type == 'point':
        experiment_method = run_point
        synthesis_method = process_point_jobs
    elif type == 'interval':
        experiment_method = run_interval
        synthesis_method = process_interval_jobs
    elif type == 'distribution':
        experiment_method = run_probabilistic
        synthesis_method = process_probabilistic_jobs

    if distributed:
        import dispy, dispy.httpd

        nodes = kwargs.get("nodes", ['127.0.0.1'])
        cluster, http_server = cUtil.start_dispy_cluster(experiment_method, nodes)

    experiments = 0
    jobs = []

    if progress:
        from tqdm import tqdm
        progressbar = tqdm(total=len(data), desc="Sliding Window:")

    inc = __pop("inc", 0.1, kwargs)

    for ct, train, test in cUtil.sliding_window(data, windowsize, train, inc=inc, **kwargs):
        experiments += 1

        if progress:
            progressbar.update(windowsize * inc)

        partitioners_pool = []

        if partitioners_models is None:

            for partition in partitions:

                for partitioner in partitioners_methods:

                    data_train_fs = partitioner(data=train, npart=partition, transformation=transformation)

                    partitioners_pool.append(data_train_fs)
        else:
            partitioners_pool = partitioners_models

        rng1 = partitioners_pool

        if progress:
            rng1 = tqdm(partitioners_pool, desc="Partitioners")

        for partitioner in rng1:

            rng2 = enumerate(pool,start=0)

            if progress:
                rng2 = enumerate(tqdm(pool, desc="Models"),start=0)

            for _id, model in rng2:

                if not distributed:
                    job = experiment_method(deepcopy(model), deepcopy(partitioner), train, test, **kwargs)
                    jobs.append(job)
                else:
                    job = cluster.submit(deepcopy(model), deepcopy(partitioner), train, test, **kwargs)
                    job.id = id  # associate an ID to identify jobs (if needed later)
                    jobs.append(job)

    if progress:
        progressbar.close()

    if distributed:
        jobs2 = []

        rng = jobs

        cluster.wait()  # wait for all jobs to finish

        if progress:
            rng = tqdm(jobs)

        for job in rng:
            if job.status == dispy.DispyJob.Finished and job is not None:
                tmp = job()
                jobs2.append(tmp)

        jobs = deepcopy(jobs2)

        cUtil.stop_dispy_cluster(cluster, http_server)

    file = kwargs.get('file', None)
    sintetic = kwargs.get('sintetic', False)

    return synthesis_method(jobs, experiments, save, file, sintetic)


def get_benchmark_point_methods():
    """Return all non FTS methods for point forecasting"""
    return [naive.Naive, arima.ARIMA, quantreg.QuantileRegression]


def get_point_methods():
    """Return all FTS methods for point forecasting"""
    return [song.ConventionalFTS, chen.ConventionalFTS, yu.WeightedFTS, ismailefendi.ImprovedWeightedFTS,
            cheng.TrendWeightedFTS, sadaei.ExponentialyWeightedFTS, hofts.HighOrderFTS, hwang.HighOrderFTS,
            pwfts.ProbabilisticWeightedFTS]


def get_benchmark_interval_methods():
    """Return all non FTS methods for point_to_interval forecasting"""
    return [quantreg.QuantileRegression]


def get_interval_methods():
    """Return all FTS methods for point_to_interval forecasting"""
    return [ifts.IntervalFTS, pwfts.ProbabilisticWeightedFTS]


def get_probabilistic_methods():
    """Return all FTS methods for probabilistic forecasting"""
    return [ensemble.AllMethodEnsembleFTS, pwfts.ProbabilisticWeightedFTS]


def get_benchmark_probabilistic_methods():
    """Return all FTS methods for probabilistic forecasting"""
    return [arima.ARIMA, quantreg.QuantileRegression]


def run_point(mfts, partitioner, train_data, test_data, window_key=None, **kwargs):
    """
    Point forecast benchmark function to be executed on cluster nodes
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results
    """
    import time
    from pyFTS.models import yu, chen, hofts, pwfts,ismailefendi,sadaei, song, cheng, hwang
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures, naive, arima, quantreg
    from pyFTS.common import Transformations

    tmp = [song.ConventionalFTS, chen.ConventionalFTS, yu.WeightedFTS, ismailefendi.ImprovedWeightedFTS,
           cheng.TrendWeightedFTS, sadaei.ExponentialyWeightedFTS, hofts.HighOrderFTS, hwang.HighOrderFTS,
           pwfts.ProbabilisticWeightedFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp4 = [naive.Naive, arima.ARIMA, quantreg.QuantileRegression]

    tmp3 = [Measures.get_point_statistics]

    tmp5 = [Transformations.Differential]

    transformation = kwargs.get('transformation', None)
    indexer = kwargs.get('indexer', None)

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "")
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner

    if transformation is not None:
        mfts.append_transformation(transformation)

    _start = time.time()
    mfts.fit(train_data, order=mfts.order, **kwargs)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    _rmse, _smape, _u = Measures.get_point_statistics(test_data, mfts, **kwargs)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times, 'window': window_key}

    return ret


def run_interval(mfts, partitioner, train_data, test_data, window_key=None, **kwargs):
    """
    Interval forecast benchmark function to be executed on cluster nodes
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results
    """
    import time
    from pyFTS.models import hofts,ifts,pwfts
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures, arima, quantreg

    tmp = [hofts.HighOrderFTS, ifts.IntervalFTS,  pwfts.ProbabilisticWeightedFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp4 = [arima.ARIMA, quantreg.QuantileRegression]

    tmp3 = [Measures.get_interval_statistics]

    transformation = kwargs.get('transformation', None)
    indexer = kwargs.get('indexer', None)

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "") + str(mfts.alpha)
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner

    if transformation is not None:
        mfts.append_transformation(transformation)

    _start = time.time()
    mfts.fit(train_data, order=mfts.order, **kwargs)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    _sharp, _res, _cov, _q05, _q25, _q75, _q95 = Measures.get_interval_statistics(test_data, mfts, **kwargs)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'sharpness': _sharp, 'resolution': _res, 'coverage': _cov, 'time': times,
           'Q05': _q05, 'Q25': _q25, 'Q75': _q75, 'Q95': _q95, 'window': window_key}

    return ret


def run_probabilistic(mfts, partitioner, train_data, test_data, window_key=None, **kwargs):
    """
    Probabilistic forecast benchmark function to be executed on cluster nodes
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param steps:
    :param resolution:
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results
    """
    import time
    import numpy as np
    from pyFTS.models import hofts, ifts, pwfts
    from pyFTS.models.ensemble import ensemble
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures, arima
    from pyFTS.models.seasonal import SeasonalIndexer

    tmp = [hofts.HighOrderFTS, ifts.IntervalFTS, pwfts.ProbabilisticWeightedFTS, arima.ARIMA, ensemble.AllMethodEnsembleFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp3 = [Measures.get_distribution_statistics, SeasonalIndexer.SeasonalIndexer, SeasonalIndexer.LinearSeasonalIndexer]

    transformation = kwargs.get('transformation', None)
    indexer = kwargs.get('indexer', None)

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "") + str(mfts.alpha)
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner

    if transformation is not None:
        mfts.append_transformation(transformation)

    if mfts.has_seasonality:
        mfts.indexer = indexer

    try:
        _start = time.time()
        mfts.fit(train_data, order=mfts.order)
        _end = time.time()
        times = _end - _start

        _crps1, _t1 = Measures.get_distribution_statistics(test_data, mfts, **kwargs)
        _t1 += times
    except Exception as e:
        print(e)
        _crps1 = np.nan
        _t1 = np.nan

    ret = {'key': _key, 'obj': mfts, 'CRPS': _crps1, 'time': _t1, 'window': window_key}

    return ret


def build_model_pool_point(models, max_order, benchmark_models, benchmark_models_parameters):
    pool = []
    if models is None:
        models = get_point_methods()
    for model in models:
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            mfts.order = 1
            pool.append(mfts)

    if benchmark_models is not None:
        for count, model in enumerate(benchmark_models, start=0):
            par = benchmark_models_parameters[count]
            mfts = model(str(par if par is not None else ""))
            mfts.order = par
            pool.append(mfts)
    return pool


def process_point_jobs(jobs, experiments, save=False, file=None, sintetic=False):
    objs = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    for job in jobs:
        _key = job['key']
        if _key not in objs:
            objs[_key] = job['obj']
            rmse[_key] = []
            smape[_key] = []
            u[_key] = []
            times[_key] = []
        rmse[_key].append(job['rmse'])
        smape[_key].append(job['smape'])
        u[_key].append(job['u'])
        times[_key].append(job['time'])

    return bUtil.save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u)


def process_interval_jobs(jobs, experiments, save=False, file=None, sintetic=False):
    objs = {}
    sharpness = {}
    resolution = {}
    coverage = {}
    q05 = {}
    q25 = {}
    q75 = {}
    q95 = {}
    times = {}

    for job in jobs:
        _key = job['key']
        if _key not in objs:
            objs[_key] = job['obj']
            sharpness[_key] = []
            resolution[_key] = []
            coverage[_key] = []
            times[_key] = []
            q05[_key] = []
            q25[_key] = []
            q75[_key] = []
            q95[_key] = []

        sharpness[_key].append(job['sharpness'])
        resolution[_key].append(job['resolution'])
        coverage[_key].append(job['coverage'])
        times[_key].append(job['time'])
        q05[_key].append(job['Q05'])
        q25[_key].append(job['Q25'])
        q75[_key].append(job['Q75'])
        q95[_key].append(job['Q95'])


    return bUtil.save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, sintetic,
                                         times, q05, q25, q75, q95)


def process_probabilistic_jobs(jobs, experiments, save=False, file=None, sintetic=False):
    objs = {}
    crps = {}
    times = {}

    for job in jobs:
        _key = job['key']
        if _key not in objs:
            objs[_key] = job['obj']
            crps[_key] = []
            times[_key] = []

        crps[_key].append(job['CRPS'])
        times[_key].append(job['time'])

    return bUtil.save_dataframe_probabilistic(experiments, file, objs, crps, times, save, sintetic)



def print_point_statistics(data, models, externalmodels = None, externalforecasts = None, indexers=None):
    ret = "Model		& Order     & RMSE		& SMAPE      & Theil's U		\\\\ \n"
    for count,model in enumerate(models,start=0):
        _rmse, _smape, _u = Measures.get_point_statistics(data, model, indexers)
        ret += model.shortname + "		& "
        ret += str(model.order) + "		& "
        ret += str(_rmse) + "		& "
        ret += str(_smape)+ "		& "
        ret += str(_u)
        #ret += str(round(Measures.TheilsInequality(np.array(data[fts.order:]), np.array(forecasts[:-1])), 4))
        ret += "	\\\\ \n"
    if externalmodels is not None:
        l = len(externalmodels)
        for k in np.arange(0,l):
            ret += externalmodels[k] + "		& "
            ret += " 1		& "
            ret += str(round(Measures.rmse(data, externalforecasts[k][:-1]), 2)) + "		& "
            ret += str(round(Measures.smape(data, externalforecasts[k][:-1]), 2))+ "		& "
            ret += str(round(Measures.UStatistic(data, externalforecasts[k][:-1]), 2))
            ret += "	\\\\ \n"
    print(ret)



def print_interval_statistics(original, models):
    ret = "Model	& Order     & Sharpness		& Resolution		& Coverage & .05  & .25 & .75 & .95	\\\\ \n"
    for fts in models:
        _sharp, _res, _cov, _q5, _q25, _q75, _q95  = Measures.get_interval_statistics(original, fts)
        ret += fts.shortname + "		& "
        ret += str(fts.order) + "		& "
        ret += str(_sharp) + "		& "
        ret += str(_res) + "		& "
        ret += str(_cov) + "        &"
        ret += str(_q5) + "        &"
        ret += str(_q25) + "        &"
        ret += str(_q75) + "        &"
        ret += str(_q95) + "\\\\ \n"
    print(ret)







def ahead_sliding_window(data, windowsize, train, steps, models=None, resolution = None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         save=False, file=None, synthetic=False):
    if models is None:
        models = [pwfts.ProbabilisticWeightedFTS]

    objs = {}
    lcolors = {}
    crps_interval = {}
    crps_distr = {}
    times1 = {}
    times2 = {}

    experiments = 0
    for ct, train,test in cUtil.sliding_window(data, windowsize, train):
        experiments += 1
        for partition in partitions:
            for partitioner in partitioners:
                pttr = str(partitioner.__module__).split('.')[-1]
                data_train_fs = partitioner(data=train, npart=partition, transformation=transformation)

                for count, model in enumerate(models, start=0):

                    mfts = model("")
                    _key = mfts.shortname + " " + pttr+ " q = " +str(partition)

                    mfts.partitioner = data_train_fs
                    if not mfts.is_high_order:

                        if dump: print(ct,_key)

                        if _key not in objs:
                            objs[_key] = mfts
                            lcolors[_key] = colors[count % ncol]
                            crps_interval[_key] = []
                            crps_distr[_key] = []
                            times1[_key] = []
                            times2[_key] = []

                        if transformation is not None:
                            mfts.append_transformation(transformation)

                        _start = time.time()
                        mfts.train(train, sets=data_train_fs.sets)
                        _end = time.time()

                        _tdiff = _end - _start

                        _crps1, _crps2, _t1, _t2 = Measures.get_distribution_statistics(test,mfts,steps=steps,resolution=resolution)

                        crps_interval[_key].append_rhs(_crps1)
                        crps_distr[_key].append_rhs(_crps2)
                        times1[_key] = _tdiff + _t1
                        times2[_key] = _tdiff + _t2

                        if dump: print(_crps1, _crps2, _tdiff, _t1, _t2)

                    else:
                        for order in np.arange(1, max_order + 1):
                            if order >= mfts.min_order:
                                mfts = model("")
                                _key = mfts.shortname + " n = " + str(order) + " " + pttr + " q = " + str(partition)
                                mfts.partitioner = data_train_fs

                                if dump: print(ct,_key)

                                if _key not in objs:
                                    objs[_key] = mfts
                                    lcolors[_key] = colors[count % ncol]
                                    crps_interval[_key] = []
                                    crps_distr[_key] = []
                                    times1[_key] = []
                                    times2[_key] = []

                                if transformation is not None:
                                    mfts.append_transformation(transformation)

                                _start = time.time()
                                mfts.train(train, sets=data_train_fs.sets, order=order)
                                _end = time.time()

                                _tdiff = _end - _start

                                _crps1, _crps2, _t1, _t2 = Measures.get_distribution_statistics(test, mfts, steps=steps,
                                                                                       resolution=resolution)

                                crps_interval[_key].append_rhs(_crps1)
                                crps_distr[_key].append_rhs(_crps2)
                                times1[_key] = _tdiff + _t1
                                times2[_key] = _tdiff + _t2

                                if dump: print(_crps1, _crps2, _tdiff, _t1, _t2)

    return bUtil.save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, synthetic)



def all_ahead_forecasters(data_train, data_test, partitions, start, steps, resolution = None, max_order=3,save=False, file=None, tam=[20, 5],
                           models=None, transformation=None, option=2):
    if models is None:
        models = [pwfts.ProbabilisticWeightedFTS]

    if resolution is None: resolution = (max(data_train) - min(data_train)) / 100

    objs = []

    data_train_fs = Grid.GridPartitioner(data=data_train, npart=partitions, transformation=transformation).sets
    lcolors = []

    for count, model in cUtil.enumerate2(models, start=0, step=2):
        mfts = model("")
        if not mfts.is_high_order:
            if transformation is not None:
                mfts.append_transformation(transformation)
            mfts.train(data_train, sets=data_train_fs.sets)
            objs.append(mfts)
            lcolors.append( colors[count % ncol] )
        else:
            for order in np.arange(1,max_order+1):
                if order >= mfts.min_order:
                    mfts = model(" n = " + str(order))
                    if transformation is not None:
                        mfts.append_transformation(transformation)
                    mfts.train(data_train, sets=data_train_fs.sets, order=order)
                    objs.append(mfts)
                    lcolors.append(colors[count % ncol])

    distributions = [False for k in objs]

    distributions[0] = True

    print_distribution_statistics(data_test[start:], objs, steps, resolution)

    plot_compared_intervals_ahead(data_test, objs, lcolors, distributions=distributions, time_from=start, time_to=steps,
                               interpol=False, save=save, file=file, tam=tam, resolution=resolution, option=option)





def print_distribution_statistics(original, models, steps, resolution):
    ret = "Model	& Order     &  Interval & Distribution	\\\\ \n"
    for fts in models:
        _crps1, _crps2, _t1, _t2 = Measures.get_distribution_statistics(original, fts, steps, resolution)
        ret += fts.shortname + "		& "
        ret += str(fts.order) + "		& "
        ret += str(_crps1) + "		& "
        ret += str(_crps2) + "	\\\\ \n"
    print(ret)



def plot_compared_intervals_ahead(original, models, colors, distributions, time_from, time_to, intervals = True,
                               save=False, file=None, tam=[20, 5], resolution=None,
                               cmap='Blues', linewidth=1.5):
    """
    Plot the forecasts of several one step ahead models, by point or by interval 
    :param original: Original time series data (list)
    :param models: List of models to compare
    :param colors: List of models colors
    :param distributions: True to plot a distribution
    :param time_from: index of data poit to start the ahead forecasting
    :param time_to: number of steps ahead to forecast
    :param interpol: Fill space between distribution plots
    :param save: Save the picture on file
    :param file: Filename to save the picture
    :param tam: Size of the picture
    :param resolution: 
    :param cmap: Color map to be used on distribution plot 
    :param option: Distribution type to be passed for models
    :return: 
    """
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    cm = plt.get_cmap(cmap)
    cNorm = pltcolors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    if resolution is None: resolution = (max(original) - min(original)) / 100

    mi = []
    ma = []

    for count, fts in enumerate(models, start=0):
        if fts.has_probability_forecasting and distributions[count]:
            density = fts.forecast_ahead_distribution(original[time_from - fts.order:time_from], time_to,
                                                      resolution=resolution)

            #plot_density_scatter(ax, cmap, density, fig, resolution, time_from, time_to)
            plot_density_rectange(ax, cm, density, fig, resolution, time_from, time_to)

        if fts.has_interval_forecasting and intervals:
            forecasts = fts.forecast_ahead_interval(original[time_from - fts.order:time_from], time_to)
            lower = [kk[0] for kk in forecasts]
            upper = [kk[1] for kk in forecasts]
            mi.append(min(lower))
            ma.append(max(upper))
            for k in np.arange(0, time_from - fts.order):
                lower.insert(0, None)
                upper.insert(0, None)
            ax.plot(lower, color=colors[count], label=fts.shortname, linewidth=linewidth)
            ax.plot(upper, color=colors[count], linewidth=linewidth*1.5)

    ax.plot(original, color='black', label="Original", linewidth=linewidth*1.5)
    handles0, labels0 = ax.get_legend_handles_labels()
    if True in distributions:
        lgd = ax.legend(handles0, labels0, loc=2)
    else:
        lgd = ax.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))
    _mi = min(mi)
    if _mi < 0:
        _mi *= 1.1
    else:
        _mi *= 0.9
    _ma = max(ma)
    if _ma < 0:
        _ma *= 0.9
    else:
        _ma *= 1.1

    ax.set_ylim([_mi, _ma])
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])

    cUtil.show_and_save_image(fig, file, save, lgd=lgd)



def plot_density_rectange(ax, cmap, density, fig, resolution, time_from, time_to):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    patches = []
    colors = []
    for x in density.index:
        for y in density.columns:
            s = Rectangle((time_from + x, y), 1, resolution, fill=True, lw = 0)
            patches.append(s)
            colors.append(density[y][x]*5)
    pc = PatchCollection(patches=patches, match_original=True)
    pc.set_clim([0, 1])
    pc.set_cmap(cmap)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label('Density')


from pyFTS.common import Transformations



def plot_distribution(ax, cmap, probabilitydist, fig, time_from, reference_data=None):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    patches = []
    colors = []
    for ct, dt in enumerate(probabilitydist):
        disp = 0.0
        if reference_data is not None:
            disp = reference_data[time_from+ct]

        for y in dt.bins:
            s = Rectangle((time_from+ct, y+disp), 1, dt.resolution, fill=True, lw = 0)
            patches.append(s)
            colors.append(dt.density(y))
    scale = Transformations.Scale()
    colors = scale.apply(colors)
    pc = PatchCollection(patches=patches, match_original=True)
    pc.set_clim([0, 1])
    pc.set_cmap(cmap)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label('Density')


def plot_interval(axis, intervals, order, label, color='red', typeonlegend=False, ls='-', linewidth=1):
    lower = [kk[0] for kk in intervals]
    upper = [kk[1] for kk in intervals]
    mi = min(lower) * 0.95
    ma = max(upper) * 1.05
    for k in np.arange(0, order):
        lower.insert(0, None)
        upper.insert(0, None)
    if typeonlegend: label += " (Interval)"
    axis.plot(lower, color=color, label=label, ls=ls,linewidth=linewidth)
    axis.plot(upper, color=color, ls=ls,linewidth=linewidth)
    return [mi, ma]



def plot_compared_series(original, models, colors, typeonlegend=False, save=False, file=None, tam=[20, 5],
                         points=True, intervals=True, linewidth=1.5):
    """
    Plot the forecasts of several one step ahead models, by point or by interval
    :param original: Original time series data (list)
    :param models: List of models to compare
    :param colors: List of models colors
    :param typeonlegend: Add the type of forecast (point / interval) on legend
    :param save: Save the picture on file
    :param file: Filename to save the picture
    :param tam: Size of the picture
    :param points: True to plot the point forecasts, False otherwise
    :param intervals: True to plot the interval forecasts, False otherwise
    :param linewidth:
    :return:
    """

    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    mi = []
    ma = []

    legends = []

    ax.plot(original, color='black', label="Original", linewidth=linewidth*1.5)

    for count, fts in enumerate(models, start=0):
        try:
            if fts.has_point_forecasting and points:
                forecasts = fts.forecast(original)
                if isinstance(forecasts, np.ndarray):
                    forecasts = forecasts.tolist()
                mi.append(min(forecasts) * 0.95)
                ma.append(max(forecasts) * 1.05)
                for k in np.arange(0, fts.order):
                    forecasts.insert(0, None)
                lbl = fts.shortname + str(fts.order if fts.is_high_order and not fts.benchmark_only else "")
                if typeonlegend: lbl += " (Point)"
                ax.plot(forecasts, color=colors[count], label=lbl, ls="-",linewidth=linewidth)

            if fts.has_interval_forecasting and intervals:
                forecasts = fts.forecast_interval(original)
                lbl = fts.shortname + " " + str(fts.order if fts.is_high_order and not fts.benchmark_only else "")
                if not points and intervals:
                    ls = "-"
                else:
                    ls = "--"
                tmpmi, tmpma = plot_interval(ax, forecasts, fts.order, label=lbl, typeonlegend=typeonlegend,
                                             color=colors[count], ls=ls, linewidth=linewidth)
                mi.append(tmpmi)
                ma.append(tmpma)
        except ValueError as ex:
            print(fts.shortname)

        handles0, labels0 = ax.get_legend_handles_labels()
        lgd = ax.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))
        legends.append(lgd)

    # ax.set_title(fts.name)
    ax.set_ylim([min(mi), max(ma)])
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])

    #Util.show_and_save_image(fig, file, save, lgd=legends)


def plot_probability_distributions(pmfs, lcolors, tam=[15, 7]):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    for k,m in enumerate(pmfs,start=0):
        m.plot(ax, color=lcolors[k])

    handles0, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles0, labels0)



def plotCompared(original, forecasts, labels, title):
    fig = plt.figure(figsize=[13, 6])
    ax = fig.add_subplot(111)
    ax.plot(original, color='k', label="Original")
    for c in range(0, len(forecasts)):
        ax.plot(forecasts[c], label=labels[c])
    handles0, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles0, labels0)
    ax.set_title(title)
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])
    ax.set_ylim([min(original), max(original)])


def SelecaoSimples_MenorRMSE(original, parameters, modelo):
    ret = []
    errors = []
    forecasted_best = []
    print("Série Original")
    fig = plt.figure(figsize=[20, 12])
    fig.suptitle("Comparação de modelos ")
    ax0 = fig.add_axes([0, 0.5, 0.65, 0.45])  # left, bottom, width, height
    ax0.set_xlim([0, len(original)])
    ax0.set_ylim([min(original), max(original)])
    ax0.set_title('Série Temporal')
    ax0.set_ylabel('F(T)')
    ax0.set_xlabel('T')
    ax0.plot(original, label="Original")
    min_rmse = 100000.0
    best = None
    for p in parameters:
        sets = Grid.GridPartitioner(data=original, npart=p).sets
        fts = modelo(str(p) + " particoes")
        fts.train(original, sets=sets)
        # print(original)
        forecasted = fts.forecast(original)
        forecasted.insert(0, original[0])
        # print(forecasted)
        ax0.plot(forecasted, label=fts.name)
        error = Measures.rmse(np.array(forecasted), np.array(original))
        print(p, error)
        errors.append(error)
        if error < min_rmse:
            min_rmse = error
            best = fts
            forecasted_best = forecasted
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0)
    ax1 = fig.add_axes([0.7, 0.5, 0.3, 0.45])  # left, bottom, width, height
    ax1.set_title('Comparação dos Erros Quadráticos Médios')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Quantidade de Partições')
    ax1.set_xlim([min(parameters), max(parameters)])
    ax1.plot(parameters, errors)
    ret.append(best)
    ret.append(forecasted_best)
    # Modelo diferencial
    print("\nSérie Diferencial")
    difffts = Transformations.differential(original)
    errors = []
    forecastedd_best = []
    ax2 = fig.add_axes([0, 0, 0.65, 0.45])  # left, bottom, width, height
    ax2.set_xlim([0, len(difffts)])
    ax2.set_ylim([min(difffts), max(difffts)])
    ax2.set_title('Série Temporal')
    ax2.set_ylabel('F(T)')
    ax2.set_xlabel('T')
    ax2.plot(difffts, label="Original")
    min_rmse = 100000.0
    bestd = None
    for p in parameters:
        sets = Grid.GridPartitioner(data=difffts, npart=p)
        fts = modelo(str(p) + " particoes")
        fts.train(difffts, sets=sets)
        forecasted = fts.forecast(difffts)
        forecasted.insert(0, difffts[0])
        ax2.plot(forecasted, label=fts.name)
        error = Measures.rmse(np.array(forecasted), np.array(difffts))
        print(p, error)
        errors.append(error)
        if error < min_rmse:
            min_rmse = error
            bestd = fts
            forecastedd_best = forecasted
    handles0, labels0 = ax2.get_legend_handles_labels()
    ax2.legend(handles0, labels0)
    ax3 = fig.add_axes([0.7, 0, 0.3, 0.45])  # left, bottom, width, height
    ax3.set_title('Comparação dos Erros Quadráticos Médios')
    ax3.set_ylabel('RMSE')
    ax3.set_xlabel('Quantidade de Partições')
    ax3.set_xlim([min(parameters), max(parameters)])
    ax3.plot(parameters, errors)
    ret.append(bestd)
    ret.append(forecastedd_best)
    return ret


def compareModelsPlot(original, models_fo, models_ho):
    fig = plt.figure(figsize=[13, 6])
    fig.suptitle("Comparação de modelos ")
    ax0 = fig.add_axes([0, 0, 1, 1])  # left, bottom, width, height
    rows = []
    for model in models_fo:
        fts = model["model"]
        ax0.plot(model["forecasted"], label=model["name"])
    for model in models_ho:
        fts = model["model"]
        ax0.plot(model["forecasted"], label=model["name"])
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0)


def compareModelsTable(original, models_fo, models_ho):
    fig = plt.figure(figsize=[12, 4])
    fig.suptitle("Comparação de modelos ")
    columns = ['Modelo', 'Ordem', 'Partições', 'RMSE', 'MAPE (%)']
    rows = []
    for model in models_fo:
        fts = model["model"]
        error_r = Measures.rmse(model["forecasted"], original)
        error_m = round(Measures.mape(model["forecasted"], original) * 100, 2)
        rows.append([model["name"], fts.order, len(fts.sets), error_r, error_m])
    for model in models_ho:
        fts = model["model"]
        error_r = Measures.rmse(model["forecasted"][fts.order:], original[fts.order:])
        error_m = round(Measures.mape(model["forecasted"][fts.order:], original[fts.order:]) * 100, 2)
        rows.append([model["name"], fts.order, len(fts.sets), error_r, error_m])
    ax1 = fig.add_axes([0, 0, 1, 1])  # left, bottom, width, height
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.table(cellText=rows,
              colLabels=columns,
              cellLoc='center',
              bbox=[0, 0, 1, 1])
    sup = "\\begin{tabular}{"
    header = ""
    body = ""
    footer = ""

    for c in columns:
        sup = sup + "|c"
        if len(header) > 0:
            header = header + " & "
        header = header + "\\textbf{" + c + "} "
    sup = sup + "|} \\hline\n"
    header = header + "\\\\ \\hline \n"

    for r in rows:
        lin = ""
        for c in r:
            if len(lin) > 0:
                lin = lin + " & "
            lin = lin + str(c)

        body = body + lin + "\\\\ \\hline \n"

    return sup + header + body + "\\end{tabular}"


def simpleSearch_RMSE(train, test, model, partitions, orders, save=False, file=None, tam=[10, 15],
                      plotforecasts=False, elev=30, azim=144, intervals=False,parameters=None,
                      partitioner=Grid.GridPartitioner,transformation=None,indexer=None):
    _3d = len(orders) > 1
    ret = []
    if _3d:
        errors = np.array([[0 for k in range(len(partitions))] for kk in range(len(orders))])
    else:
        errors = []
    forecasted_best = []
    fig = plt.figure(figsize=tam)
    # fig.suptitle("Comparação de modelos ")
    if plotforecasts:
        ax0 = fig.add_axes([0, 0.4, 0.9, 0.5])  # left, bottom, width, height
        ax0.set_xlim([0, len(train)])
        ax0.set_ylim([min(train) * 0.9, max(train) * 1.1])
        ax0.set_title('Forecasts')
        ax0.set_ylabel('F(T)')
        ax0.set_xlabel('T')
    min_rmse = 1000000.0
    best = None

    for pc, p in enumerate(partitions, start=0):

        sets = partitioner(data=train, npart=p, transformation=transformation).sets
        for oc, o in enumerate(orders, start=0):
            fts = model("q = " + str(p) + " n = " + str(o))
            fts.append_transformation(transformation)
            fts.train(train, sets=sets, order=o, parameters=parameters)
            if not intervals:
                forecasted = fts.forecast(test)
                if not fts.has_seasonality:
                    error = Measures.rmse(np.array(test[o:]), np.array(forecasted[:-1]))
                else:
                    error = Measures.rmse(np.array(test[o:]), np.array(forecasted))
                for kk in range(o):
                    forecasted.insert(0, None)
                if plotforecasts: ax0.plot(forecasted, label=fts.name)
            else:
                forecasted = fts.forecast_interval(test)
                error = 1.0 - Measures.rmse_interval(np.array(test[o:]), np.array(forecasted[:-1]))
            if _3d:
                errors[oc, pc] = error
            else:
                errors.append( error )
            if error < min_rmse:
                min_rmse = error
                best = fts
                forecasted_best = forecasted

    # print(min_rmse)
    if plotforecasts:
        # handles0, labels0 = ax0.get_legend_handles_labels()
        # ax0.legend(handles0, labels0)
        ax0.plot(test, label="Original", linewidth=3.0, color="black")
        if _3d: ax1 = Axes3D(fig, rect=[0, 1, 0.9, 0.9], elev=elev, azim=azim)
    if _3d and not plotforecasts:
        ax1 = Axes3D(fig, rect=[0, 1, 0.9, 0.9], elev=elev, azim=azim)
        ax1.set_title('Error Surface')
        ax1.set_ylabel('Model order')
        ax1.set_xlabel('Number of partitions')
        ax1.set_zlabel('RMSE')
        X, Y = np.meshgrid(partitions, orders)
        surf = ax1.plot_surface(X, Y, errors, rstride=1, cstride=1, antialiased=True)
    else:
        ax1 = fig.add_axes([0, 1, 0.9, 0.9])
        ax1.set_title('Error Curve')
        ax1.set_xlabel('Number of partitions')
        ax1.set_ylabel('RMSE')
        ax1.plot(partitions, errors)
    ret.append(best)
    ret.append(forecasted_best)
    ret.append(min_rmse)

    # plt.tight_layout()

    cUtil.show_and_save_image(fig, file, save)

    return ret



def pftsExploreOrderAndPartitions(data,save=False, file=None):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=[6, 8])
    data_fs1 = Grid.GridPartitioner(data=data, npart=10).sets
    mi = []
    ma = []

    axes[0].set_title('Point Forecasts by Order')
    axes[2].set_title('Interval Forecasts by Order')

    for order in np.arange(1, 6):
        fts = pwfts.ProbabilisticWeightedFTS("")
        fts.shortname = "n = " + str(order)
        fts.train(data, sets=data_fs1.sets, order=order)
        point_forecasts = fts.forecast(data)
        interval_forecasts = fts.forecast_interval(data)
        lower = [kk[0] for kk in interval_forecasts]
        upper = [kk[1] for kk in interval_forecasts]
        mi.append(min(lower) * 0.95)
        ma.append(max(upper) * 1.05)
        for k in np.arange(0, order):
            point_forecasts.insert(0, None)
            lower.insert(0, None)
            upper.insert(0, None)
        axes[0].plot(point_forecasts, label=fts.shortname)
        axes[2].plot(lower, label=fts.shortname)
        axes[2].plot(upper)

    axes[1].set_title('Point Forecasts by Number of Partitions')
    axes[3].set_title('Interval Forecasts by Number of Partitions')

    for partitions in np.arange(5, 11):
        data_fs = Grid.GridPartitioner(data=data, npart=partitions).sets
        fts = pwfts.ProbabilisticWeightedFTS("")
        fts.shortname = "q = " + str(partitions)
        fts.train(data, sets=data_fs.sets, order=1)
        point_forecasts = fts.forecast(data)
        interval_forecasts = fts.forecast_interval(data)
        lower = [kk[0] for kk in interval_forecasts]
        upper = [kk[1] for kk in interval_forecasts]
        mi.append(min(lower) * 0.95)
        ma.append(max(upper) * 1.05)
        point_forecasts.insert(0, None)
        lower.insert(0, None)
        upper.insert(0, None)
        axes[1].plot(point_forecasts, label=fts.shortname)
        axes[3].plot(lower, label=fts.shortname)
        axes[3].plot(upper)

    for ax in axes:
        ax.set_ylabel('F(T)')
        ax.set_xlabel('T')
        ax.plot(data, label="Original", color="black", linewidth=1.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))
        ax.set_ylim([min(mi), max(ma)])
        ax.set_xlim([0, len(data)])

    plt.tight_layout()

    cUtil.show_and_save_image(fig, file, save)

