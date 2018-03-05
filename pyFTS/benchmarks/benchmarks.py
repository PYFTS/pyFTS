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
from pyFTS.common import Util
# from sklearn.cross_validation import KFold
from pyFTS.partitioners import Grid
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

colors = ['grey', 'darkgrey', 'rosybrown', 'maroon', 'red','orange', 'gold', 'yellow', 'olive', 'green',
          'darkgreen', 'cyan', 'lightblue','blue', 'darkblue', 'purple', 'darkviolet' ]

ncol = len(colors)

styles = ['-','--','-.',':','.']

nsty = len(styles)


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
    return [arima.ARIMA, ensemble.AllMethodEnsembleFTS, pwfts.ProbabilisticWeightedFTS]


def run_point(mfts, partitioner, train_data, test_data, window_key=None, transformation=None, indexer=None):
    """
    Point forecast benchmark function to be executed on sliding window
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "")
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner
        if transformation is not None:
            mfts.append_transformation(transformation)

    _start = time.time()
    mfts.train(train_data, sets=partitioner.sets, order=mfts.order)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    _rmse, _smape, _u = Measures.get_point_statistics(test_data, mfts, indexer)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times, 'window': window_key}

    return ret


def point_sliding_window(data, windowsize, train=0.8, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         benchmark_models=None, benchmark_models_parameters = None,
                         save=False, file=None, sintetic=False):
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
    :param benchmark_models: Non FTS models to benchmark
    :param benchmark_models_parameters: Non FTS models parameters
    :param save: save results
    :param file: file path to save the results
    :param sintetic: if true only the average and standard deviation of the results
    :return: DataFrame with the results
    """

    if benchmark_models is None: # and models is None:
        benchmark_models = [naive.Naive, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA,
                            quantreg.QuantileRegression, quantreg.QuantileRegression]

    if benchmark_models_parameters is None:
        benchmark_models_parameters = [1, (1, 0, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2), 1, 2]

    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    pool = []
    jobs = []
    objs = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

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

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train):
        experiments += 1

        benchmarks_only = {}

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(data=train, npart=partition, transformation=transformation)

                for _id, m in enumerate(pool,start=0):
                    if m.benchmark_only and m.shortname in benchmarks_only:
                        continue
                    else:
                        benchmarks_only[m.shortname] = m

                    tmp = run_point(deepcopy(m), data_train_fs, train, test, ct, transformation)

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
                    print(tmp['key'], tmp['window'])

    _process_end = time.time()

    print("Process End: {0: %H:%M:%S}".format(datetime.datetime.now()))

    print("Process Duration: {0}".format(_process_end - _process_start))

    return bUtil.save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u)


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


def all_point_forecasters(data_train, data_test, partitions, max_order=3, statistics=True, residuals=True,
                        series=True, save=False, file=None, tam=[20, 5], models=None, transformation=None,
                        distributions=False, benchmark_models=None, benchmark_models_parameters=None):
    """
    Fixed data benchmark for FTS point forecasters
    :param data_train: data used to train the models
    :param data_test: data used to test the models
    :param partitions: the max number of partitions on the Universe of Discourse 
    :param max_order: the max order of the models (for high order models)
    :param statistics: print statistics
    :param residuals: print and plot residuals
    :param series: plot time series
    :param save: save results
    :param file: file path to save the results
    :param tam: figure dimensions to plot the graphs 
    :param models: list of models to benchmark
    :param transformation: data transformation
    :param distributions: plot distributions
    :return: 
    """
    models = build_model_pool_point(models, max_order, benchmark_models, benchmark_models_parameters)

    objs = []

    data_train_fs = Grid.GridPartitioner(data=data_train, npart=partitions, transformation=transformation)

    count = 1

    lcolors = []

    for count, model in enumerate(models, start=0):
        #print(model)
        if transformation is not None:
            model.append_transformation(transformation)
        model.train(data_train, sets=data_train_fs.sets, order=model.order)
        objs.append(model)
        lcolors.append( colors[count % ncol] )

    if statistics:
        print_point_statistics(data_test, objs)

    if residuals:
        print(ResidualAnalysis.compare_residuals(data_test, objs))
        ResidualAnalysis.plot_residuals(data_test, objs, save=save, file=file, tam=tam)

    if series:
        plot_compared_series(data_test, objs, lcolors, typeonlegend=False, save=save, file=file, tam=tam,
                             intervals=False)

    if distributions:
        lcolors.insert(0,'black')
        pmfs = []
        pmfs.append(
            ProbabilityDistribution.ProbabilityDistribution("Original", 100, [min(data_test), max(data_test)], data=data_test) )

        for m in objs:
            forecasts = m.forecast(data_test)
            pmfs.append(
                ProbabilityDistribution.ProbabilityDistribution(m.shortname, 100, [min(data_test), max(data_test)],
                                                                data=forecasts))
        print(getProbabilityDistributionStatistics(pmfs,data_test))

        plot_probability_distributions(pmfs, lcolors, tam=tam)

    return models


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



def getProbabilityDistributionStatistics(pmfs, data):
    ret = "Model		& Entropy     & Empirical Likelihood		&  Pseudo Likelihood		\\\\ \n"
    for k in pmfs:
        ret += k.name + "		& "
        ret += str(k.entropy()) + "		& "
        ret += str(k.empiricalloglikelihood())+ "		& "
        ret += str(k.pseudologlikelihood(data))
        ret += "	\\\\ \n"
    return ret



def interval_sliding_window(data, windowsize, train=0.8, models=None, partitioners=[Grid.GridPartitioner],
                            partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                            save=False, file=None, synthetic=True):
    if models is None:
        models = get_interval_methods()

    objs = {}
    lcolors = {}
    sharpness = {}
    resolution = {}
    coverage = {}
    times = {}

    experiments = 0
    for ct, training,test in Util.sliding_window(data, windowsize, train):
        experiments += 1
        for partition in partitions:
            for partitioner in partitioners:
                pttr = str(partitioner.__module__).split('.')[-1]
                data_train_fs = partitioner(data=training, npart=partition, transformation=transformation)

                for count, model in enumerate(models, start=0):

                    mfts = model("")
                    _key = mfts.shortname + " " + pttr+ " q = " +str(partition)

                    mfts.partitioner = data_train_fs
                    if not mfts.is_high_order:

                        if dump: print(ct,_key)

                        if _key not in objs:
                            objs[_key] = mfts
                            lcolors[_key] = colors[count % ncol]
                            sharpness[_key] = []
                            resolution[_key] = []
                            coverage[_key] = []
                            times[_key] = []

                        if transformation is not None:
                            mfts.append_transformation(transformation)

                        _start = time.time()
                        mfts.train(training, sets=data_train_fs.sets)
                        _end = time.time()
                        _tdiff = _end - _start

                        _start = time.time()
                        _sharp, _res, _cov = Measures.get_interval_statistics(test, mfts)
                        _end = time.time()
                        _tdiff += _end - _start
                        sharpness[_key].append_rhs(_sharp)
                        resolution[_key].append_rhs(_res)
                        coverage[_key].append_rhs(_cov)
                        times[_key].append_rhs(_tdiff)

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
                                    sharpness[_key] = []
                                    resolution[_key] = []
                                    coverage[_key] = []
                                    times[_key] = []

                                if transformation is not None:
                                    mfts.append_transformation(transformation)

                                _start = time.time()
                                mfts.train(training, sets=data_train_fs.sets, order=order)
                                _end = time.time()

                                _tdiff = _end - _start

                                _start = time.time()
                                _sharp, _res, _cov = Measures.get_interval_statistics(test, mfts)
                                _end = time.time()
                                _tdiff += _end - _start
                                sharpness[_key].append_rhs(_sharp)
                                resolution[_key].append_rhs(_res)
                                coverage[_key].append_rhs(_cov)
                                times[_key].append_rhs(_tdiff)

    return bUtil.save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, synthetic, times)


def build_model_pool_interval(models, max_order, benchmark_models, benchmark_models_parameters):
    pool = []
    if models is None:
        models = get_interval_methods()
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
    alphas = [0.05, 0.25]
    if benchmark_models is not None:
        for count, model in enumerate(benchmark_models, start=0):
            par = benchmark_models_parameters[count]
            for alpha in alphas:
                mfts = model(str(alpha), alpha=alpha)
                mfts.order = par
                pool.append(mfts)
    return pool


def all_interval_forecasters(data_train, data_test, partitions, max_order=3,save=False, file=None, tam=[20, 5],
                             statistics=False, models=None, transformation=None,
                             benchmark_models=None, benchmark_models_parameters=None):
    models = build_model_pool_interval(models, max_order, benchmark_models, benchmark_models_parameters)

    data_train_fs = Grid.GridPartitioner(data=data_train, npart=partitions, transformation=transformation).sets

    lcolors = []
    objs = []

    for count, model in Util.enumerate2(models, start=0, step=2):
        if transformation is not None:
            model.append_transformation(transformation)
            model.train(data_train, sets=data_train_fs, order=model.order)
        objs.append(model)
        lcolors.append( colors[count % ncol] )

    if statistics:
        print_interval_statistics(data_test, objs)

    plot_compared_series(data_test, objs, lcolors, typeonlegend=False, save=save, file=file, tam=tam,
                         points=False, intervals=True)


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
    for ct, train,test in Util.sliding_window(data, windowsize, train):
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

    for count, model in Util.enumerate2(models, start=0, step=2):
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

    Util.show_and_save_image(fig, file, save, lgd=lgd)


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


def plot_probabilitydistribution_density(ax, cmap, probabilitydist, fig, time_from):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    patches = []
    colors = []
    for ct, dt in enumerate(probabilitydist):
        for y in dt.bins:
            s = Rectangle((time_from+ct, y), 1, dt.resolution, fill=True, lw = 0)
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

    Util.show_and_save_image(fig, file, save)

    return ret


def sliding_window_simple_search(data, windowsize, model, partitions, orders, save=False, file=None, tam=[10, 15],
                                 plotforecasts=False, elev=30, azim=144, intervals=False, parameters=None):
    _3d = len(orders) > 1
    ret = []
    errors = np.array([[0 for k in range(len(partitions))] for kk in range(len(orders))])
    forecasted_best = []
    fig = plt.figure(figsize=tam)
    # fig.suptitle("Comparação de modelos ")
    if plotforecasts:
        ax0 = fig.add_axes([0, 0.4, 0.9, 0.5])  # left, bottom, width, height
        ax0.set_xlim([0, len(data)])
        ax0.set_ylim([min(data) * 0.9, max(data) * 1.1])
        ax0.set_title('Forecasts')
        ax0.set_ylabel('F(T)')
        ax0.set_xlabel('T')
    min_rmse = 1000000.0
    best = None

    for pc, p in enumerate(partitions, start=0):

        sets = Grid.GridPartitioner(data=data, npart=p).sets
        for oc, o in enumerate(orders, start=0):
            _error = []
            for ct, train, test in Util.sliding_window(data, windowsize, 0.8):
                fts = model("q = " + str(p) + " n = " + str(o))
                fts.train(data, sets=sets, order=o, parameters=parameters)
                if not intervals:
                    forecasted = fts.forecast(test)
                    if not fts.has_seasonality:
                        _error.append( Measures.rmse(np.array(test[o:]), np.array(forecasted[:-1])) )
                    else:
                        _error.append( Measures.rmse(np.array(test[o:]), np.array(forecasted)) )
                    for kk in range(o):
                        forecasted.insert(0, None)
                    if plotforecasts: ax0.plot(forecasted, label=fts.name)
                else:
                    forecasted = fts.forecast_interval(test)
                    _error.append( 1.0 - Measures.rmse_interval(np.array(test[o:]), np.array(forecasted[:-1])) )
            error = np.nanmean(_error)
            errors[oc, pc] = error
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
    if not plotforecasts: ax1 = Axes3D(fig, rect=[0, 1, 0.9, 0.9], elev=elev, azim=azim)
    # ax1 = fig.add_axes([0.6, 0.5, 0.45, 0.45], projection='3d')
    if _3d:
        ax1.set_title('Error Surface')
        ax1.set_ylabel('Model order')
        ax1.set_xlabel('Number of partitions')
        ax1.set_zlabel('RMSE')
        X, Y = np.meshgrid(partitions, orders)
        surf = ax1.plot_surface(X, Y, errors, rstride=1, cstride=1, antialiased=True)
    else:
        ax1 = fig.add_axes([0, 1, 0.9, 0.9])
        ax1.set_title('Error Curve')
        ax1.set_ylabel('Number of partitions')
        ax1.set_xlabel('RMSE')
        ax0.plot(errors,partitions)
    ret.append(best)
    ret.append(forecasted_best)

    # plt.tight_layout()

    Util.show_and_save_image(fig, file, save)

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

    Util.show_and_save_image(fig, file, save)

