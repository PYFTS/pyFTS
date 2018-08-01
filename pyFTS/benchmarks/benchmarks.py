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
from pyFTS.common import Transformations
from pyFTS.models import song, chen, yu, ismailefendi, sadaei, hofts, pwfts, ifts, cheng, hwang
from pyFTS.models.ensemble import ensemble
from pyFTS.benchmarks import Measures, naive, arima, ResidualAnalysis, quantreg, knn
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
    return [ arima.ARIMA, quantreg.QuantileRegression]


def get_interval_methods():
    """Return all FTS methods for point_to_interval forecasting"""
    return [ifts.IntervalFTS, pwfts.ProbabilisticWeightedFTS]


def get_probabilistic_methods():
    """Return all FTS methods for probabilistic forecasting"""
    return [ensemble.AllMethodEnsembleFTS, pwfts.ProbabilisticWeightedFTS]


def get_benchmark_probabilistic_methods():
    """Return all FTS methods for probabilistic forecasting"""
    return [arima.ARIMA, quantreg.QuantileRegression, knn.KNearestNeighbors]


def sliding_window_benchmarks(data, windowsize, train=0.8, **kwargs):
    """
    Sliding window benchmarks for FTS forecasters.

    For each data window, a train and test datasets will be splitted. For each train split, number of
    partitions and partitioning method will be created a partitioner model. And for each partitioner, order,
    steps ahead and FTS method a foreasting model will be trained.

    Then all trained models are benchmarked on the test data and the metrics are stored on a sqlite3 database
    (identified by the 'file' parameter) for posterior analysis.

    All these process can be distributed on a dispy cluster, setting the atributed 'distributed' to true and
    informing the list of dispy nodes on 'nodes' parameter.

    The number of experiments is determined by 'windowsize' and 'inc' parameters.

    :param data: test data
    :param windowsize: size of sliding window
    :param train: percentual of sliding window data used to train the models
    :param kwargs: dict, optional arguments

    :keyword
        benchmark_methods:  a list with Non FTS models to benchmark. The default is None.
        benchmark_methods_parameters:  a list with Non FTS models parameters. The default is None.
        benchmark_models: A boolean value indicating if external FTS methods will be used on benchmark. The default is False.
        build_methods: A boolean value indicating if the default FTS methods will be used on benchmark. The default is True.
        dataset: the dataset name to identify the current set of benchmarks results on database.
        distributed: A boolean value indicating if the forecasting procedure will be distributed in a dispy cluster. . The default is False
        file: file path to save the results. The default is benchmarks.db.
        inc: a float on interval [0,1] indicating the percentage of the windowsize to move the window
        methods: a list with FTS class names. The default depends on the forecasting type and contains the list of all FTS methods.
        models: a list with prebuilt FTS objects. The default is None.
        nodes: a list with the dispy cluster nodes addresses. The default is [127.0.0.1].
        orders: a list with orders of the models (for high order models). The default is [1,2,3].
        partitions: a list with the numbers of partitions on the Universe of Discourse. The default is [10].
        partitioners_models: a list with prebuilt Universe of Discourse partitioners objects. The default is None.
        partitioners_methods: a list with Universe of Discourse partitioners class names. The default is [partitioners.Grid.GridPartitioner].
        progress: If true a progress bar will be displayed during the benchmarks. The default is False.
        start: in the multi step forecasting, the index of the data where to start forecasting. The default is 0.
        steps_ahead: a list with  the forecasting horizons, i. e., the number of steps ahead to forecast. The default is 1.
        tag: a name to identify the current set of benchmarks results on database.
        type: the forecasting type, one of these values: point(default), interval or distribution. The default is point.
        transformations: a list with data transformations do apply . The default is [None].
    """

    tag = __pop('tag', None, kwargs)
    dataset = __pop('dataset', None, kwargs)

    distributed = __pop('distributed', False, kwargs)

    transformations = kwargs.get('transformations', [None])
    progress = kwargs.get('progress', None)
    type = kwargs.get("type", 'point')

    orders = __pop("orders", [1,2,3], kwargs)

    partitioners_models = __pop("partitioners_models", None, kwargs)
    partitioners_methods = __pop("partitioners_methods", [Grid.GridPartitioner], kwargs)
    partitions = __pop("partitions", [10], kwargs)

    steps_ahead = __pop('steps_ahead', [1], kwargs)

    methods = __pop('methods', None, kwargs)

    models = __pop('models', None, kwargs)

    pool = [] if models is None else models

    if methods is None:
        if type  == 'point':
            methods = get_point_methods()
        elif type == 'interval':
            methods = get_interval_methods()
        elif type == 'distribution':
            methods = get_probabilistic_methods()

    build_methods = __pop("build_methods", True, kwargs)

    if build_methods:
        for method in methods:
            mfts = method()

            if mfts.is_high_order:
                for order in orders:
                    if order >= mfts.min_order:
                        mfts = method()
                        mfts.order = order
                        pool.append(mfts)
            else:
                mfts.order = 1
                pool.append(mfts)

    benchmark_models = __pop("benchmark_models", False, kwargs)

    if benchmark_models != False:

        benchmark_methods = __pop("benchmark_methods", None, kwargs)
        benchmark_methods_parameters = __pop("benchmark_methods_parameters", None, kwargs)

        benchmark_pool = [] if ( benchmark_models is None or not isinstance(benchmark_models, list)) \
            else benchmark_models

        if benchmark_models is None and benchmark_methods is None:
            if type == 'point'or type  == 'partition':
                benchmark_methods = get_benchmark_point_methods()
            elif type == 'interval':
                benchmark_methods = get_benchmark_interval_methods()
            elif type == 'distribution':
                benchmark_methods = get_benchmark_probabilistic_methods()

        if benchmark_methods is not None:
            for transformation in transformations:
                for count, model in enumerate(benchmark_methods, start=0):
                    par = benchmark_methods_parameters[count]
                    mfts = model(**par)
                    mfts.append_transformation(transformation)
                    benchmark_pool.append(mfts)

    if type == 'point':
        experiment_method = run_point
        synthesis_method = process_point_jobs
    elif type == 'interval':
        experiment_method = run_interval
        synthesis_method = process_interval_jobs
    elif type == 'distribution':
        experiment_method = run_probabilistic
        synthesis_method = process_probabilistic_jobs
    else:
        raise ValueError("Type parameter has a unkown value!")

    if distributed:
        import dispy, dispy.httpd

        nodes = kwargs.get("nodes", ['127.0.0.1'])
        cluster, http_server = cUtil.start_dispy_cluster(experiment_method, nodes)

    experiments = 0
    jobs = []

    inc = __pop("inc", 0.1, kwargs)

    if progress:
        from tqdm import tqdm
        _tdata = len(data) / (windowsize * inc)
        _tasks = (len(partitioners_models) * len(orders) * len(partitions) * len(transformations) * len(steps_ahead))
        _tbcmk = len(benchmark_pool)*len(steps_ahead)
        progressbar = tqdm(total=_tdata*_tasks + _tdata*_tbcmk, desc="Benchmarks:")

    file = kwargs.get('file', "benchmarks.db")

    conn = bUtil.open_benchmark_db(file)

    for ct, train, test in cUtil.sliding_window(data, windowsize, train, inc=inc, **kwargs):
        if benchmark_models != False:
            for model in benchmark_pool:
                for step in steps_ahead:

                    kwargs['steps_ahead'] = step

                    if not distributed:
                        if progress:
                            progressbar.update(1)
                        job = experiment_method(deepcopy(model), None, train, test, **kwargs)
                        synthesis_method(dataset, tag, job, conn)
                    else:
                        job = cluster.submit(deepcopy(model), None, train, test, **kwargs)
                        jobs.append(job)

        partitioners_pool = []

        if partitioners_models is None:

            for transformation in transformations:

                for partition in partitions:

                    for partitioner in partitioners_methods:
                        print(transformation, partition)

                        data_train_fs = partitioner(data=train, npart=partition, transformation=transformation)

                        partitioners_pool.append(data_train_fs)
        else:
            partitioners_pool = partitioners_models

        for step in steps_ahead:

            for partitioner in partitioners_pool:

                for _id, model in enumerate(pool,start=0):

                    kwargs['steps_ahead'] = step

                    if not distributed:
                        if progress:
                            progressbar.update(1)
                        job = experiment_method(deepcopy(model), deepcopy(partitioner), train, test, **kwargs)
                        synthesis_method(dataset, tag, job, conn)
                    else:
                        job = cluster.submit(deepcopy(model), deepcopy(partitioner), train, test, **kwargs)
                        job.id = id  # associate an ID to identify jobs (if needed later)
                        jobs.append(job)

    if progress:
        progressbar.close()

    if distributed:

        for job in jobs:
            if progress:
                progressbar.update(1)
            job()
            if job.status == dispy.DispyJob.Finished and job is not None:
                tmp = job.result
                synthesis_method(dataset, tag, tmp, conn)
            else:
                print("status",job.status)
                print("result",job.result)
                print("stdout",job.stdout)
                print("stderr",job.exception)

        cluster.wait()  # wait for all jobs to finish

        cUtil.stop_dispy_cluster(cluster, http_server)

    conn.close()




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

    indexer = kwargs.get('indexer', None)

    steps_ahead = kwargs.get('steps_ahead', 1)
    method = kwargs.get('method', None)

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "")
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner
        mfts.append_transformation(partitioner.transformation)

    _key += str(steps_ahead)
    _key += str(method) if method is not None else ""

    _start = time.time()
    mfts.fit(train_data, **kwargs)
    _end = time.time()
    times = _end - _start


    _start = time.time()
    _rmse, _smape, _u = Measures.get_point_statistics(test_data, mfts, **kwargs)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times, 'window': window_key,
           'steps': steps_ahead, 'method': method}

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

    steps_ahead = kwargs.get('steps_ahead', 1)
    method = kwargs.get('method', None)

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "") + str(mfts.alpha)
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner
        mfts.append_transformation(partitioner.transformation)

    _key += str(steps_ahead)
    _key += str(method) if method is not None else ""

    _start = time.time()
    mfts.fit(train_data, **kwargs)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    #_sharp, _res, _cov, _q05, _q25, _q75, _q95, _w05, _w25
    metrics = Measures.get_interval_statistics(test_data, mfts, **kwargs)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'sharpness': metrics[0], 'resolution': metrics[1], 'coverage': metrics[2],
           'time': times,'Q05': metrics[3], 'Q25': metrics[4], 'Q75': metrics[5], 'Q95': metrics[6],
           'winkler05': metrics[7], 'winkler25': metrics[8],
           'window': window_key,'steps': steps_ahead, 'method': method}

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
    from pyFTS.benchmarks import Measures, arima, quantreg, knn
    from pyFTS.models.seasonal import SeasonalIndexer

    tmp = [hofts.HighOrderFTS, ifts.IntervalFTS, pwfts.ProbabilisticWeightedFTS, arima.ARIMA,
           ensemble.AllMethodEnsembleFTS, knn.KNearestNeighbors]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp3 = [Measures.get_distribution_statistics, SeasonalIndexer.SeasonalIndexer, SeasonalIndexer.LinearSeasonalIndexer]

    indexer = kwargs.get('indexer', None)

    steps_ahead = kwargs.get('steps_ahead', 1)
    method = kwargs.get('method', None)

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "") + str(mfts.alpha)
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner
        mfts.append_transformation(partitioner.transformation)

    _key += str(steps_ahead)
    _key += str(method) if method is not None else ""

    if mfts.has_seasonality:
        mfts.indexer = indexer

    _start = time.time()
    mfts.fit(train_data, **kwargs)
    _end = time.time()
    times = _end - _start

    _crps1, _t1, _brier = Measures.get_distribution_statistics(test_data, mfts, **kwargs)
    _t1 += times

    ret = {'key': _key, 'obj': mfts, 'CRPS': _crps1, 'time': _t1, 'brier': _brier, 'window': window_key,
           'steps': steps_ahead, 'method': method}

    return ret


def process_point_jobs(dataset, tag,  job, conn):

    data = bUtil.process_common_data(dataset, tag, 'point',job)

    rmse = deepcopy(data)
    rmse.extend(["rmse", job["rmse"]])
    bUtil.insert_benchmark(rmse, conn)
    smape = deepcopy(data)
    smape.extend(["smape", job["smape"]])
    bUtil.insert_benchmark(smape, conn)
    u = deepcopy(data)
    u.extend(["u", job["u"]])
    bUtil.insert_benchmark(u, conn)
    time = deepcopy(data)
    time.extend(["time", job["time"]])
    bUtil.insert_benchmark(time, conn)


def process_interval_jobs(dataset, tag, job, conn):

    data = bUtil.process_common_data(dataset, tag, 'interval', job)

    sharpness = deepcopy(data)
    sharpness.extend(["sharpness", job["sharpness"]])
    bUtil.insert_benchmark(sharpness, conn)
    resolution = deepcopy(data)
    resolution.extend(["resolution", job["resolution"]])
    bUtil.insert_benchmark(resolution, conn)
    coverage = deepcopy(data)
    coverage.extend(["coverage", job["coverage"]])
    bUtil.insert_benchmark(coverage, conn)
    time = deepcopy(data)
    time.extend(["time", job["time"]])
    bUtil.insert_benchmark(time, conn)
    Q05 = deepcopy(data)
    Q05.extend(["Q05", job["Q05"]])
    bUtil.insert_benchmark(Q05, conn)
    Q25 = deepcopy(data)
    Q25.extend(["Q25", job["Q25"]])
    bUtil.insert_benchmark(Q25, conn)
    Q75 = deepcopy(data)
    Q75.extend(["Q75", job["Q75"]])
    bUtil.insert_benchmark(Q75, conn)
    Q95 = deepcopy(data)
    Q95.extend(["Q95", job["Q95"]])
    bUtil.insert_benchmark(Q95, conn)
    W05 = deepcopy(data)
    W05.extend(["winkler05", job["winkler05"]])
    bUtil.insert_benchmark(W05, conn)
    W25 = deepcopy(data)
    W25.extend(["winkler25", job["winkler25"]])
    bUtil.insert_benchmark(W25, conn)


def process_probabilistic_jobs(dataset, tag,  job, conn):

    data = bUtil.process_common_data(dataset, tag,  'density', job)

    crps = deepcopy(data)
    crps.extend(["crps",job["CRPS"]])
    bUtil.insert_benchmark(crps, conn)
    time = deepcopy(data)
    time.extend(["time", job["time"]])
    bUtil.insert_benchmark(time, conn)
    brier = deepcopy(data)
    brier.extend(["brier", job["brier"]])
    bUtil.insert_benchmark(brier, conn)


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


def plot_point(axis, points, order, label, color='red', ls='-', linewidth=1):
    mi = min(points) * 0.95
    ma = max(points) * 1.05
    for k in np.arange(0, order):
        points.insert(0, None)
    axis.plot(points, color=color, label=label, ls=ls,linewidth=linewidth)
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

