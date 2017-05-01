#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import time
import datetime
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cross_validation import KFold
from pyFTS.partitioners import partitioner, Grid, Huarng, Entropy, FCM
from pyFTS.benchmarks import Measures, naive, arima, ResidualAnalysis, ProbabilityDistribution, Util
from pyFTS.common import Membership, FuzzySet, FLR, Transformations, Util
from pyFTS import fts, chen, yu, ismailefendi, sadaei, hofts, hwang,  pwfts, ifts
from copy import deepcopy

colors = ['grey', 'rosybrown', 'maroon', 'red','orange', 'yellow', 'olive', 'green',
          'cyan', 'blue', 'darkblue', 'purple', 'darkviolet']

ncol = len(colors)

styles = ['-','--','-.',':','.']

nsty = len(styles)

def get_benchmark_point_methods():
    return [naive.Naive, arima.ARIMA]

def get_point_methods():
    return [chen.ConventionalFTS, yu.WeightedFTS, ismailefendi.ImprovedWeightedFTS,
                  sadaei.ExponentialyWeightedFTS, hofts.HighOrderFTS, pwfts.ProbabilisticWeightedFTS]

def get_interval_methods():
    return [ifts.IntervalFTS, pwfts.ProbabilisticWeightedFTS]


def external_point_sliding_window(models, parameters, data, windowsize,train=0.8, dump=False, save=False, file=None, sintetic=True):
    objs = {}
    lcolors = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train):
        experiments += 1
        for count, method in enumerate(models, start=0):
            model = method("")

            _start = time.time()
            model.train(train, None, parameters=parameters[count])
            _end = time.time()

            _key = model.shortname

            if dump: print(ct, _key)

            if _key not in objs:
                objs[_key] = model
                lcolors[_key] = colors[count % ncol]
                rmse[_key] = []
                smape[_key] = []
                u[_key] = []
                times[_key] = []

            _tdiff = _end - _start

            try:
                _start = time.time()
                _rmse, _smape, _u = Measures.get_point_statistics(test, model, None)
                _end = time.time()
                rmse[_key].append(_rmse)
                smape[_key].append(_smape)
                u[_key].append(_u)
                _tdiff += _end - _start
                times[_key].append(_tdiff)
                if dump: print(_rmse, _smape, _u, _tdiff)
            except:
                rmse[_key].append(np.nan)
                smape[_key].append(np.nan)
                u[_key].append(np.nan)
                times[_key].append(np.nan)

    return Util.save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u)


def point_sliding_window(data, windowsize, train=0.8,models=None,partitioners=[Grid.GridPartitioner],
                   partitions=[10], max_order=3,transformation=None,indexer=None,dump=False,
                   save=False, file=None, sintetic=True):

    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    if models is None:
        models = get_point_methods()


    objs = {}
    lcolors = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    experiments = 0
    for ct, train,test in Util.sliding_window(data, windowsize, train):
        experiments += 1
        for partition in partitions:
            for partitioner in partitioners:
                pttr = str(partitioner.__module__).split('.')[-1]
                data_train_fs = partitioner(train, partition, transformation=transformation)

                for count, model in enumerate(models, start=0):

                    mfts = model("")

                    _key = mfts.shortname + " " + pttr + " q = " + str(partition)

                    mfts.partitioner = data_train_fs
                    if not mfts.isHighOrder:

                        if dump: print(ct,_key)

                        if _key not in objs:
                            objs[_key] = mfts
                            lcolors[_key] = colors[count % ncol]
                            rmse[_key] = []
                            smape[_key] = []
                            u[_key] = []
                            times[_key] = []

                        if transformation is not None:
                            mfts.appendTransformation(transformation)


                        _start = time.time()
                        mfts.train(train, data_train_fs.sets)
                        _end = time.time()
                        times[_key].append(_end - _start)

                        _start = time.time()
                        _rmse, _smape, _u = Measures.get_point_statistics(test, mfts, indexer)
                        _end = time.time()
                        rmse[_key].append(_rmse)
                        smape[_key].append(_smape)
                        u[_key].append(_u)
                        times[_key].append(_end - _start)

                        if dump: print(_rmse, _smape, _u)

                    else:
                        for order in np.arange(1, max_order + 1):
                            if order >= mfts.minOrder:
                                mfts = model("")

                                _key = mfts.shortname + " n = " + str(order) + " " + pttr + " q = " + str(partition)

                                mfts.partitioner = data_train_fs

                                if dump: print(ct,_key)

                                if _key not in objs:
                                    objs[_key] = mfts
                                    lcolors[_key] = colors[count % ncol]
                                    rmse[_key] = []
                                    smape[_key] = []
                                    u[_key] = []
                                    times[_key] = []

                                if transformation is not None:
                                    mfts.appendTransformation(transformation)

                                try:
                                    _start = time.time()
                                    mfts.train(train, data_train_fs.sets, order=order)
                                    _end = time.time()
                                    times[_key].append(_end - _start)

                                    _start = time.time()
                                    _rmse, _smape, _u = Measures.get_point_statistics(test, mfts, indexer)
                                    _end = time.time()
                                    rmse[_key].append(_rmse)
                                    smape[_key].append(_smape)
                                    u[_key].append(_u)
                                    times[_key].append(_end - _start)

                                    if dump: print(_rmse, _smape, _u)

                                except Exception as e:
                                    print(e)
                                    rmse[_key].append(np.nan)
                                    smape[_key].append(np.nan)
                                    u[_key].append(np.nan)
                                    times[_key].append(np.nan)

    _process_end = time.time()

    print("Process End: {0: %H:%M:%S}".format(datetime.datetime.now()))

    print("Process Duration: {0}".format(_process_end - _process_start))

    return Util.save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u)


def all_point_forecasters(data_train, data_test, partitions, max_order=3, statistics=True, residuals=True,
                        series=True, save=False, file=None, tam=[20, 5], models=None, transformation=None,
                        distributions=False):

    if models is None:
        models = get_point_methods()

    objs = []

    data_train_fs = Grid.GridPartitioner(data_train, partitions, transformation=transformation)

    count = 1

    lcolors = []

    for count, model in enumerate(models, start=0):
        #print(model)
        mfts = model("")
        if not mfts.isHighOrder:
            if transformation is not None:
                mfts.appendTransformation(transformation)
            mfts.train(data_train, data_train_fs.sets)
            objs.append(mfts)
            lcolors.append( colors[count % ncol] )
        else:
            for order in np.arange(1,max_order+1):
                if order >= mfts.minOrder:
                    mfts = model(" n = " + str(order))
                    if transformation is not None:
                        mfts.appendTransformation(transformation)
                    mfts.train(data_train, data_train_fs.sets, order=order)
                    objs.append(mfts)
                    lcolors.append(colors[(count + order) % ncol])

    if statistics:
        print_point_statistics(data_test, objs)

    if residuals:
        print(ResidualAnalysis.compareResiduals(data_test, objs))
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

def save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, sintetic, times):
    ret = []
    if sintetic:
        for k in sorted(objs.keys()):
            mod = []
            mfts = objs[k]
            mod.append(mfts.shortname)
            mod.append(mfts.order)
            mod.append(mfts.partitioner.name)
            mod.append(mfts.partitioner.partitions)
            mod.append(round(np.nanmean(sharpness[k]), 2))
            mod.append(round(np.nanstd(sharpness[k]), 2))
            mod.append(round(np.nanmean(resolution[k]), 2))
            mod.append(round(np.nanstd(resolution[k]), 2))
            mod.append(round(np.nanmean(coverage[k]), 2))
            mod.append(round(np.nanstd(coverage[k]), 2))
            mod.append(round(np.nanmean(times[k]), 2))
            mod.append(round(np.nanstd(times[k]), 2))
            mod.append(len(mfts))
            ret.append(mod)

        columns = ["Model", "Order", "Scheme", "Partitions", "SHARPAVG", "SHARPSTD", "RESAVG", "RESSTD", "COVAVG",
                   "COVSTD", "TIMEAVG", "TIMESTD", "SIZE"]
    else:
        for k in sorted(objs.keys()):
            try:
                mfts = objs[k]
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),
                       'Sharpness']
                tmp.extend(sharpness[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),
                       'Resolution']
                tmp.extend(resolution[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),
                       'Coverage']
                tmp.extend(coverage[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),
                       'TIME']
                tmp.extend(times[k])
                ret.append(deepcopy(tmp))
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)
        columns = [str(k) for k in np.arange(0, experiments)]
        columns.insert(0, "Model")
        columns.insert(1, "Order")
        columns.insert(2, "Scheme")
        columns.insert(3, "Partitions")
        columns.insert(4, "Size")
        columns.insert(5, "Measure")
    dat = pd.DataFrame(ret, columns=columns)
    if save: dat.to_csv(Util.uniquefilename(file), sep=";")
    return dat


def interval_sliding_window(data, windowsize, train=0.8,models=None,partitioners=[Grid.GridPartitioner],
                   partitions=[10], max_order=3,transformation=None,indexer=None,dump=False,
                   save=False, file=None, sintetic=True):
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
                data_train_fs = partitioner(training, partition, transformation=transformation)

                for count, model in enumerate(models, start=0):

                    mfts = model("")
                    _key = mfts.shortname + " " + pttr+ " q = " +str(partition)

                    mfts.partitioner = data_train_fs
                    if not mfts.isHighOrder:

                        if dump: print(ct,_key)

                        if _key not in objs:
                            objs[_key] = mfts
                            lcolors[_key] = colors[count % ncol]
                            sharpness[_key] = []
                            resolution[_key] = []
                            coverage[_key] = []
                            times[_key] = []

                        if transformation is not None:
                            mfts.appendTransformation(transformation)

                        _start = time.time()
                        mfts.train(training, data_train_fs.sets)
                        _end = time.time()
                        _tdiff = _end - _start

                        _start = time.time()
                        _sharp, _res, _cov = Measures.get_interval_statistics(test, mfts)
                        _end = time.time()
                        _tdiff += _end - _start
                        sharpness[_key].append(_sharp)
                        resolution[_key].append(_res)
                        coverage[_key].append(_cov)
                        times[_key].append(_tdiff)

                    else:
                        for order in np.arange(1, max_order + 1):
                            if order >= mfts.minOrder:
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
                                    mfts.appendTransformation(transformation)

                                _start = time.time()
                                mfts.train(training, data_train_fs.sets, order=order)
                                _end = time.time()

                                _tdiff = _end - _start

                                _start = time.time()
                                _sharp, _res, _cov = Measures.get_interval_statistics(test, mfts)
                                _end = time.time()
                                _tdiff += _end - _start
                                sharpness[_key].append(_sharp)
                                resolution[_key].append(_res)
                                coverage[_key].append(_cov)
                                times[_key].append(_tdiff)

    return save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, sintetic, times)


def all_interval_forecasters(data_train, data_test, partitions, max_order=3,save=False, file=None, tam=[20, 5],
                           models=None, transformation=None):
    if models is None:
        models = get_interval_methods()

    objs = []

    data_train_fs = Grid.GridPartitioner(data_train,partitions, transformation=transformation).sets

    lcolors = []

    for count, model in Util.enumerate2(models, start=0, step=2):
        mfts = model("")
        if not mfts.isHighOrder:
            if transformation is not None:
                mfts.appendTransformation(transformation)
            mfts.train(data_train, data_train_fs)
            objs.append(mfts)
            lcolors.append( colors[count % ncol] )
        else:
            for order in np.arange(1,max_order+1):
                if order >= mfts.minOrder:
                    mfts = model(" n = " + str(order))
                    if transformation is not None:
                        mfts.appendTransformation(transformation)
                    mfts.train(data_train, data_train_fs, order=order)
                    objs.append(mfts)
                    lcolors.append(colors[count % ncol])

    print_interval_statistics(data_test, objs)

    plot_compared_series(data_test, objs, lcolors, typeonlegend=False, save=save, file=file, tam=tam, intervals=True)


def print_interval_statistics(original, models):
    ret = "Model	& Order     & Sharpness		& Resolution		& Coverage	\\\\ \n"
    for fts in models:
        _sharp, _res, _cov = Measures.get_interval_statistics(original, fts)
        ret += fts.shortname + "		& "
        ret += str(fts.order) + "		& "
        ret += str(_sharp) + "		& "
        ret += str(_res) + "		& "
        ret += str(_cov) + "	\\\\ \n"
    print(ret)


def plot_distribution(dist):
    for k in dist.index:
        alpha = np.array([dist[x][k] for x in dist]) * 100
        x = [k for x in np.arange(0, len(alpha))]
        y = dist.columns
        plt.scatter(x, y, c=alpha, marker='s', linewidths=0, cmap='Oranges', norm=pltcolors.Normalize(vmin=0, vmax=1),
                    vmin=0, vmax=1, edgecolors=None)


def plot_compared_series(original, models, colors, typeonlegend=False, save=False, file=None, tam=[20, 5],
                         points=True, intervals=True, linewidth=1.5):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    mi = []
    ma = []

    legends = []

    ax.plot(original, color='black', label="Original", linewidth=linewidth*1.5)

    for count, fts in enumerate(models, start=0):
        if fts.hasPointForecasting and points:
            forecasted = fts.forecast(original)
            mi.append(min(forecasted) * 0.95)
            ma.append(max(forecasted) * 1.05)
            for k in np.arange(0, fts.order):
                forecasted.insert(0, None)
            lbl = fts.shortname
            if typeonlegend: lbl += " (Point)"
            ax.plot(forecasted, color=colors[count], label=lbl, ls="-",linewidth=linewidth)

        if fts.hasIntervalForecasting and intervals:
            forecasted = fts.forecastInterval(original)
            lower = [kk[0] for kk in forecasted]
            upper = [kk[1] for kk in forecasted]
            mi.append(min(lower) * 0.95)
            ma.append(max(upper) * 1.05)
            for k in np.arange(0, fts.order):
                lower.insert(0, None)
                upper.insert(0, None)
            lbl = fts.shortname
            if typeonlegend: lbl += " (Interval)"
            ax.plot(lower, color=colors[count], label=lbl, ls="--",linewidth=linewidth)
            ax.plot(upper, color=colors[count], ls="--",linewidth=linewidth)

        handles0, labels0 = ax.get_legend_handles_labels()
        lgd = ax.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))
        legends.append(lgd)

    # ax.set_title(fts.name)
    ax.set_ylim([min(mi), max(ma)])
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])

    Util.showAndSaveImage(fig, file, save, lgd=legends)


def plot_probability_distributions(pmfs, lcolors, tam=[15, 7]):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    for k,m in enumerate(pmfs,start=0):
        m.plot(ax, color=lcolors[k])

    handles0, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles0, labels0)


def save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, sintetic):
    ret = []

    if sintetic:

        for k in sorted(objs.keys()):
            try:
                ret = []
                for k in sorted(objs.keys()):
                    try:
                        mod = []
                        mfts = objs[k]
                        mod.append(mfts.shortname)
                        mod.append(mfts.order)
                        mod.append(mfts.partitioner.name)
                        mod.append(mfts.partitioner.partitions)
                        mod.append(np.round(np.nanmean(crps_interval[k]), 2))
                        mod.append(np.round(np.nanstd(crps_interval[k]), 2))
                        mod.append(np.round(np.nanmean(crps_distr[k]), 2))
                        mod.append(np.round(np.nanstd(crps_distr[k]), 2))
                        mod.append(len(mfts))
                        mod.append(np.round(np.nanmean(times1[k]), 4))
                        mod.append(np.round(np.nanmean(times2[k]), 4))
                        ret.append(mod)
                    except Exception as e:
                        print('Erro: %s' % e)
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)

        columns = ["Model", "Order", "Scheme", "Partitions", "CRPS1AVG", "CRPS1STD", "CRPS2AVG", "CRPS2STD",
                   "SIZE", "TIME1AVG", "TIME2AVG"]
    else:
        for k in sorted(objs.keys()):
            try:
                mfts = objs[k]
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts), 'CRPS_Interval']
                tmp.extend(crps_interval[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),  'CRPS_Distribution']
                tmp.extend(crps_distr[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),  'TIME_Interval']
                tmp.extend(times1[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),  'TIME_Distribution']
                tmp.extend(times2[k])
                ret.append(deepcopy(tmp))
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)
        columns = [str(k) for k in np.arange(0, experiments)]
        columns.insert(0, "Model")
        columns.insert(1, "Order")
        columns.insert(2, "Scheme")
        columns.insert(3, "Partitions")
        columns.insert(4, "Size")
        columns.insert(5, "Measure")
    dat = pd.DataFrame(ret, columns=columns)
    if save: dat.to_csv(Util.uniquefilename(file), sep=";")
    return dat


def ahead_sliding_window(data, windowsize, train, steps, models=None, resolution = None, partitioners=[Grid.GridPartitioner],
                   partitions=[10], max_order=3,transformation=None,indexer=None,dump=False,
                   save=False, file=None, sintetic=False):
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
                data_train_fs = partitioner(train, partition, transformation=transformation)

                for count, model in enumerate(models, start=0):

                    mfts = model("")
                    _key = mfts.shortname + " " + pttr+ " q = " +str(partition)

                    mfts.partitioner = data_train_fs
                    if not mfts.isHighOrder:

                        if dump: print(ct,_key)

                        if _key not in objs:
                            objs[_key] = mfts
                            lcolors[_key] = colors[count % ncol]
                            crps_interval[_key] = []
                            crps_distr[_key] = []
                            times1[_key] = []
                            times2[_key] = []

                        if transformation is not None:
                            mfts.appendTransformation(transformation)

                        _start = time.time()
                        mfts.train(train, data_train_fs.sets)
                        _end = time.time()

                        _tdiff = _end - _start

                        _crps1, _crps2, _t1, _t2 = get_distribution_statistics(test,mfts,steps=steps,resolution=resolution)

                        crps_interval[_key].append(_crps1)
                        crps_distr[_key].append(_crps2)
                        times1[_key] = _tdiff + _t1
                        times2[_key] = _tdiff + _t2

                        if dump: print(_crps1, _crps2, _tdiff, _t1, _t2)

                    else:
                        for order in np.arange(1, max_order + 1):
                            if order >= mfts.minOrder:
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
                                    mfts.appendTransformation(transformation)

                                _start = time.time()
                                mfts.train(train, data_train_fs.sets, order=order)
                                _end = time.time()

                                _tdiff = _end - _start

                                _crps1, _crps2, _t1, _t2 = get_distribution_statistics(test, mfts, steps=steps,
                                                                                       resolution=resolution)

                                crps_interval[_key].append(_crps1)
                                crps_distr[_key].append(_crps2)
                                times1[_key] = _tdiff + _t1
                                times2[_key] = _tdiff + _t2

                                if dump: print(_crps1, _crps2, _tdiff, _t1, _t2)

    return save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, sintetic)


def all_ahead_forecasters(data_train, data_test, partitions, start, steps, resolution = None, max_order=3,save=False, file=None, tam=[20, 5],
                           models=None, transformation=None, option=2):
    if models is None:
        models = [pwfts.ProbabilisticWeightedFTS]

    if resolution is None: resolution = (max(data_train) - min(data_train)) / 100

    objs = []

    data_train_fs = Grid.GridPartitioner(data_train, partitions, transformation=transformation).sets
    lcolors = []

    for count, model in Util.enumerate2(models, start=0, step=2):
        mfts = model("")
        if not mfts.isHighOrder:
            if transformation is not None:
                mfts.appendTransformation(transformation)
            mfts.train(data_train, data_train_fs)
            objs.append(mfts)
            lcolors.append( colors[count % ncol] )
        else:
            for order in np.arange(1,max_order+1):
                if order >= mfts.minOrder:
                    mfts = model(" n = " + str(order))
                    if transformation is not None:
                        mfts.appendTransformation(transformation)
                    mfts.train(data_train, data_train_fs, order=order)
                    objs.append(mfts)
                    lcolors.append(colors[count % ncol])

    distributions = [False for k in objs]

    distributions[0] = True

    print_distribution_statistics(data_test[start:], objs, steps, resolution)

    plot_compared_intervals_ahead(data_test, objs, lcolors, distributions=distributions, time_from=start, time_to=steps,
                               interpol=False, save=save, file=file, tam=tam, resolution=resolution, option=option)


def get_distribution_statistics(original, model, steps, resolution):
    ret = list()
    try:
        _s1 = time.time()
        densities1 = model.forecastAheadDistribution(original, steps, parameters=3)
        _e1 = time.time()
        ret.append(round(Measures.crps(original, densities1), 3))
        ret.append(round(_e1 - _s1, 3))
    except Exception as e:
        print('Erro: ', e)
        ret.append(np.nan)
        ret.append(np.nan)

    try:
        _s2 = time.time()
        densities2 = model.forecastAheadDistribution(original, steps, parameters=2)
        _e2 = time.time()
        ret.append( round(Measures.crps(original, densities2), 3))
        ret.append(round(_e2 - _s2, 3))
    except:
        ret.append(np.nan)
        ret.append(np.nan)

    return ret


def print_distribution_statistics(original, models, steps, resolution):
    ret = "Model	& Order     &  Interval & Distribution	\\\\ \n"
    for fts in models:
        _crps1, _crps2, _t1, _t2 = get_distribution_statistics(original, fts, steps, resolution)
        ret += fts.shortname + "		& "
        ret += str(fts.order) + "		& "
        ret += str(_crps1) + "		& "
        ret += str(_crps2) + "	\\\\ \n"
    print(ret)


def plot_compared_intervals_ahead(original, models, colors, distributions, time_from, time_to,
                               interpol=False, save=False, file=None, tam=[20, 5], resolution=None,
                               cmap='Blues',option=2):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    cm = plt.get_cmap(cmap)
    cNorm = pltcolors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    if resolution is None: resolution = (max(original) - min(original)) / 100

    mi = []
    ma = []

    for count, fts in enumerate(models, start=0):
        if fts.hasDistributionForecasting and distributions[count]:
            density = fts.forecastAheadDistribution(original[time_from - fts.order:time_from], time_to,
                                                    resolution=resolution, method=option)

            Y = []
            X = []
            C = []
            S = []
            y = density.columns
            t = len(y)

            ss = time_to ** 2

            for k in density.index:
                #alpha = [scalarMap.to_rgba(density[col][k]) for col in density.columns]
                col = [density[col][k]*5 for col in density.columns]

                x = [time_from + k for x in np.arange(0, t)]

                s = [ss for x in np.arange(0, t)]

                ic = resolution/10

                for cc in np.arange(0, resolution, ic):
                    Y.append(y + cc)
                    X.append(x)
                    C.append(col)
                    S.append(s)

            Y = np.hstack(Y)
            X = np.hstack(X)
            C = np.hstack(C)
            S = np.hstack(S)

            s = ax.scatter(X, Y, c=C, marker='s',s=S, linewidths=0, edgecolors=None, cmap=cmap)
            s.set_clim([0, 1])
            cb = fig.colorbar(s)

            cb.set_label('Density')


        if fts.hasIntervalForecasting:
            forecasts = fts.forecastAheadInterval(original[time_from - fts.order:time_from], time_to)
            lower = [kk[0] for kk in forecasts]
            upper = [kk[1] for kk in forecasts]
            mi.append(min(lower))
            ma.append(max(upper))
            for k in np.arange(0, time_from - fts.order):
                lower.insert(0, None)
                upper.insert(0, None)
            ax.plot(lower, color=colors[count], label=fts.shortname)
            ax.plot(upper, color=colors[count])

        else:
            forecasts = fts.forecast(original)
            mi.append(min(forecasts))
            ma.append(max(forecasts))
            for k in np.arange(0, time_from):
                forecasts.insert(0, None)
            ax.plot(forecasts, color=colors[count], label=fts.shortname)

    ax.plot(original, color='black', label="Original")
    handles0, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles0, labels0, loc=2)
    # ax.set_title(fts.name)
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

    #plt.colorbar()

    Util.showAndSaveImage(fig, file, save)


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
        sets = Grid.GridPartitionerTrimf(original, p)
        fts = modelo(str(p) + " particoes")
        fts.train(original, sets)
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
        sets = Grid.GridPartitionerTrimf(difffts, p)
        fts = modelo(str(p) + " particoes")
        fts.train(difffts, sets)
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
    errors = np.array([[0 for k in range(len(partitions))] for kk in range(len(orders))])
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

        sets = partitioner(train, p, transformation=transformation).sets
        for oc, o in enumerate(orders, start=0):
            fts = model("q = " + str(p) + " n = " + str(o))
            fts.appendTransformation(transformation)
            fts.train(train, sets, o, parameters=parameters)
            if not intervals:
                forecasted = fts.forecast(test)
                if not fts.hasSeasonality:
                    error = Measures.rmse(np.array(test[o:]), np.array(forecasted[:-1]))
                else:
                    error = Measures.rmse(np.array(test[o:]), np.array(forecasted))
                for kk in range(o):
                    forecasted.insert(0, None)
                if plotforecasts: ax0.plot(forecasted, label=fts.name)
            else:
                forecasted = fts.forecastInterval(test)
                error = 1.0 - Measures.rmse_interval(np.array(test[o:]), np.array(forecasted[:-1]))
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
    ret.append(min_rmse)

    # plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)

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

        sets = Grid.GridPartitioner(data, p).sets
        for oc, o in enumerate(orders, start=0):
            _error = []
            for ct, train, test in Util.sliding_window(data, windowsize, 0.8):
                fts = model("q = " + str(p) + " n = " + str(o))
                fts.train(data, sets, o, parameters=parameters)
                if not intervals:
                    forecasted = fts.forecast(test)
                    if not fts.hasSeasonality:
                        _error.append( Measures.rmse(np.array(test[o:]), np.array(forecasted[:-1])) )
                    else:
                        _error.append( Measures.rmse(np.array(test[o:]), np.array(forecasted)) )
                    for kk in range(o):
                        forecasted.insert(0, None)
                    if plotforecasts: ax0.plot(forecasted, label=fts.name)
                else:
                    forecasted = fts.forecastInterval(test)
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

    Util.showAndSaveImage(fig, file, save)

    return ret


def pftsExploreOrderAndPartitions(data,save=False, file=None):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=[6, 8])
    data_fs1 = Grid.GridPartitionerTrimf(data, 10)
    mi = []
    ma = []

    axes[0].set_title('Point Forecasts by Order')
    axes[2].set_title('Interval Forecasts by Order')

    for order in np.arange(1, 6):
        fts = pwfts.ProbabilisticWeightedFTS("")
        fts.shortname = "n = " + str(order)
        fts.train(data, data_fs1, order=order)
        point_forecasts = fts.forecast(data)
        interval_forecasts = fts.forecastInterval(data)
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
        data_fs = Grid.GridPartitionerTrimf(data, partitions)
        fts = pwfts.ProbabilisticWeightedFTS("")
        fts.shortname = "q = " + str(partitions)
        fts.train(data, data_fs, 1)
        point_forecasts = fts.forecast(data)
        interval_forecasts = fts.forecastInterval(data)
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

    Util.showAndSaveImage(fig, file, save)

