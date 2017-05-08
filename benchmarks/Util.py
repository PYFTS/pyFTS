"""
Benchmark utility functions
"""

import matplotlib as plt
import matplotlib.cm as cmx
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from checkbox_support.parsers.tests.test_modinfo import testMultipleModinfoParser
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import pandas as pd
from copy import deepcopy
from pyFTS.common import Util


def extract_measure(dataframe,measure,data_columns):
    if not dataframe.empty:
        tmp = dataframe[(dataframe.Measure == measure)][data_columns].to_dict(orient="records")[0]
        ret = [k for k in tmp.values()]
        return ret
    else:
        return None


def find_best(dataframe, criteria, ascending):
    models = dataframe.Model.unique()
    orders = dataframe.Order.unique()
    ret = {}
    for m in models:
        for o in orders:
            mod = {}
            df = dataframe[(dataframe.Model == m) & (dataframe.Order == o)].sort_values(by=criteria, ascending=ascending)
            if not df.empty:
                _key = str(m) + str(o)
                best = df.loc[df.index[0]]
                mod['Model'] = m
                mod['Order'] = o
                mod['Scheme'] = best["Scheme"]
                mod['Partitions'] = best["Partitions"]

                ret[_key] = mod

    return ret


def point_dataframe_sintetic_columns():
    return ["Model", "Order", "Scheme", "Partitions", "Size", "RMSEAVG", "RMSESTD", "SMAPEAVG", "SMAPESTD", "UAVG",
            "USTD", "TIMEAVG", "TIMESTD"]


def point_dataframe_analytic_columns(experiments):
    columns = [str(k) for k in np.arange(0, experiments)]
    columns.insert(0, "Model")
    columns.insert(1, "Order")
    columns.insert(2, "Scheme")
    columns.insert(3, "Partitions")
    columns.insert(4, "Size")
    columns.insert(5, "Measure")
    return columns


def save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u):
    """
    Create a dataframe to store the benchmark results
    :param experiments: dictionary with the execution results
    :param file: 
    :param objs: 
    :param rmse: 
    :param save: 
    :param sintetic: 
    :param smape: 
    :param times: 
    :param u: 
    :return: 
    """
    ret = []

    if sintetic:

        for k in sorted(objs.keys()):
            try:
                mod = []
                mfts = objs[k]
                mod.append(mfts.shortname)
                mod.append(mfts.order)
                if not mfts.benchmark_only:
                    mod.append(mfts.partitioner.name)
                    mod.append(mfts.partitioner.partitions)
                    mod.append(len(mfts))
                else:
                    mod.append('-')
                    mod.append('-')
                    mod.append('-')
                mod.append(np.round(np.nanmean(rmse[k]), 2))
                mod.append(np.round(np.nanstd(rmse[k]), 2))
                mod.append(np.round(np.nanmean(smape[k]), 2))
                mod.append(np.round(np.nanstd(smape[k]), 2))
                mod.append(np.round(np.nanmean(u[k]), 2))
                mod.append(np.round(np.nanstd(u[k]), 2))
                mod.append(np.round(np.nanmean(times[k]), 4))
                mod.append(np.round(np.nanstd(times[k]), 4))
                ret.append(mod)
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)

        columns = point_dataframe_sintetic_columns()
    else:
        for k in sorted(objs.keys()):
            try:
                mfts = objs[k]
                n = mfts.shortname
                o = mfts.order
                if not mfts.benchmark_only:
                    s = mfts.partitioner.name
                    p = mfts.partitioner.partitions
                    l = len(mfts)
                else:
                    s = '-'
                    p = '-'
                    l = '-'

                tmp = [n, o, s, p, l, 'RMSE']
                tmp.extend(rmse[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'SMAPE']
                tmp.extend(smape[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'U']
                tmp.extend(u[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'TIME']
                tmp.extend(times[k])
                ret.append(deepcopy(tmp))
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)
        columns = point_dataframe_analytic_columns(experiments)
    dat = pd.DataFrame(ret, columns=columns)
    if save: dat.to_csv(Util.uniquefilename(file), sep=";", index=False)
    return dat


def cast_dataframe_to_sintetic_point(infile, outfile, experiments):
    columns = point_dataframe_analytic_columns(experiments)
    dat = pd.read_csv(infile, sep=";", usecols=columns)
    models = dat.Model.unique()
    orders = dat.Order.unique()
    schemes = dat.Scheme.unique()
    partitions = dat.Partitions.unique()

    data_columns = analytical_data_columns(experiments)

    ret = []

    for m in models:
        for o in orders:
            for s in schemes:
                for p in partitions:
                    mod = []
                    df = dat[(dat.Model == m) & (dat.Order == o) & (dat.Scheme == s) & (dat.Partitions == p)]
                    if not df.empty:
                        rmse = extract_measure(df, 'RMSE', data_columns)
                        smape = extract_measure(df, 'SMAPE', data_columns)
                        u = extract_measure(df, 'U', data_columns)
                        times = extract_measure(df, 'TIME', data_columns)
                        mod.append(m)
                        mod.append(o)
                        mod.append(s)
                        mod.append(p)
                        mod.append(extract_measure(df, 'RMSE', ['Size'])[0])
                        mod.append(np.round(np.nanmean(rmse), 2))
                        mod.append(np.round(np.nanstd(rmse), 2))
                        mod.append(np.round(np.nanmean(smape), 2))
                        mod.append(np.round(np.nanstd(smape), 2))
                        mod.append(np.round(np.nanmean(u), 2))
                        mod.append(np.round(np.nanstd(u), 2))
                        mod.append(np.round(np.nanmean(times), 4))
                        mod.append(np.round(np.nanstd(times), 4))
                        ret.append(mod)

    dat = pd.DataFrame(ret, columns=point_dataframe_sintetic_columns())
    dat.to_csv(Util.uniquefilename(outfile), sep=";", index=False)


def analytical_data_columns(experiments):
    data_columns = [str(k) for k in np.arange(0, experiments)]
    return data_columns


def plot_dataframe_point(file_synthetic, file_analytic, experiments):

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=[6, 8])

    axes[0].set_title('RMSE')
    axes[1].set_title('SMAPE')
    axes[2].set_title('U Statistic')
    axes[3].set_title('Execution Time')

    dat_syn = pd.read_csv(file_synthetic, sep=";", usecols=point_dataframe_sintetic_columns())

    bests = find_best(dat_syn, ['UAVG','RMSEAVG','USTD','RMSESTD'], [1,1,1,1])

    dat_ana = pd.read_csv(file_analytic, sep=";", usecols=point_dataframe_analytic_columns(experiments))

    data_columns = analytical_data_columns(experiments)

    rmse = []
    smape = []
    u = []
    times = []
    labels = []

    for b in bests.keys():
        best = bests[b]
        tmp = dat_ana[(dat_ana.Model == best["Model"]) & (dat_ana.Order == best["Order"])
                & (dat_ana.Scheme == best["Scheme"]) & (dat_ana.Partitions == best["Partitions"])]
        rmse.append( extract_measure(tmp,'RMSE',data_columns) )
        smape.append(extract_measure(tmp, 'SMAPE', data_columns))
        u.append(extract_measure(tmp, 'U', data_columns))
        times.append(extract_measure(tmp, 'TIME', data_columns))
        labels.append(best["Model"] + " " + str(best["Order"]))

    axes[0].boxplot(rmse, labels=labels, showmeans=True)
    axes[1].boxplot(smape, labels=labels, showmeans=True)
    axes[2].boxplot(u, labels=labels, showmeans=True)
    axes[3].boxplot(times, labels=labels, showmeans=True)

    plt.show()



def save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, sintetic, times):
    ret = []
    if sintetic:
        for k in sorted(objs.keys()):
            mod = []
            mfts = objs[k]
            mod.append(mfts.shortname)
            mod.append(mfts.order)
            if not mfts.benchmark_only:
                mod.append(mfts.partitioner.name)
                mod.append(mfts.partitioner.partitions)
                l = len(mfts)
            else:
                mod.append('-')
                mod.append('-')
                l = '-'
            mod.append(round(np.nanmean(sharpness[k]), 2))
            mod.append(round(np.nanstd(sharpness[k]), 2))
            mod.append(round(np.nanmean(resolution[k]), 2))
            mod.append(round(np.nanstd(resolution[k]), 2))
            mod.append(round(np.nanmean(coverage[k]), 2))
            mod.append(round(np.nanstd(coverage[k]), 2))
            mod.append(round(np.nanmean(times[k]), 2))
            mod.append(round(np.nanstd(times[k]), 2))
            mod.append(l)
            ret.append(mod)

        columns = interval_dataframe_sintetic_columns()
    else:
        for k in sorted(objs.keys()):
            try:
                mfts = objs[k]
                n = mfts.shortname
                o = mfts.order
                if not mfts.benchmark_only:
                    s = mfts.partitioner.name
                    p = mfts.partitioner.partitions
                    l = len(mfts)
                else:
                    s = '-'
                    p = '-'
                    l = '-'

                tmp = [n, o, s, p, l, 'Sharpness']
                tmp.extend(sharpness[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'Resolution']
                tmp.extend(resolution[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'Coverage']
                tmp.extend(coverage[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'TIME']
                tmp.extend(times[k])
                ret.append(deepcopy(tmp))
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)
        columns = interval_dataframe_analytic_columns(experiments)
    dat = pd.DataFrame(ret, columns=columns)
    if save: dat.to_csv(Util.uniquefilename(file), sep=";")
    return dat

def interval_dataframe_analytic_columns(experiments):
    columns = [str(k) for k in np.arange(0, experiments)]
    columns.insert(0, "Model")
    columns.insert(1, "Order")
    columns.insert(2, "Scheme")
    columns.insert(3, "Partitions")
    columns.insert(4, "Size")
    columns.insert(5, "Measure")
    return columns


def interval_dataframe_sintetic_columns():
    columns = ["Model", "Order", "Scheme", "Partitions", "SHARPAVG", "SHARPSTD", "RESAVG", "RESSTD", "COVAVG",
               "COVSTD", "TIMEAVG", "TIMESTD", "SIZE"]
    return columns


def save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, sintetic):
    """
    Save benchmark results for m-step ahead probabilistic forecasters 
    :param experiments: 
    :param file: 
    :param objs: 
    :param crps_interval: 
    :param crps_distr: 
    :param times1: 
    :param times2: 
    :param save: 
    :param sintetic: 
    :return: 
    """
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
                        if not mfts.benchmark_only:
                            mod.append(mfts.partitioner.name)
                            mod.append(mfts.partitioner.partitions)
                            l = len(mfts)
                        else:
                            mod.append('-')
                            mod.append('-')
                            l = '-'
                        mod.append(np.round(np.nanmean(crps_interval[k]), 2))
                        mod.append(np.round(np.nanstd(crps_interval[k]), 2))
                        mod.append(np.round(np.nanmean(crps_distr[k]), 2))
                        mod.append(np.round(np.nanstd(crps_distr[k]), 2))
                        mod.append(l)
                        mod.append(np.round(np.nanmean(times1[k]), 4))
                        mod.append(np.round(np.nanmean(times2[k]), 4))
                        ret.append(mod)
                    except Exception as e:
                        print('Erro: %s' % e)
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)

        columns = ahead_dataframe_sintetic_columns()
    else:
        for k in sorted(objs.keys()):
            try:
                mfts = objs[k]
                n = mfts.shortname
                o = mfts.order
                if not mfts.benchmark_only:
                    s = mfts.partitioner.name
                    p = mfts.partitioner.partitions
                    l = len(mfts)
                else:
                    s = '-'
                    p = '-'
                    l = '-'
                tmp = [n, o, s, p, l, 'CRPS_Interval']
                tmp.extend(crps_interval[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'CRPS_Distribution']
                tmp.extend(crps_distr[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'TIME_Interval']
                tmp.extend(times1[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'TIME_Distribution']
                tmp.extend(times2[k])
                ret.append(deepcopy(tmp))
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)
        columns = ahead_dataframe_analytic_columns(experiments)
    dat = pd.DataFrame(ret, columns=columns)
    if save: dat.to_csv(Util.uniquefilename(file), sep=";")
    return dat


def ahead_dataframe_analytic_columns(experiments):
    columns = [str(k) for k in np.arange(0, experiments)]
    columns.insert(0, "Model")
    columns.insert(1, "Order")
    columns.insert(2, "Scheme")
    columns.insert(3, "Partitions")
    columns.insert(4, "Size")
    columns.insert(5, "Measure")
    return columns


def ahead_dataframe_sintetic_columns():
    columns = ["Model", "Order", "Scheme", "Partitions", "CRPS1AVG", "CRPS1STD", "CRPS2AVG", "CRPS2STD",
               "SIZE", "TIME1AVG", "TIME2AVG"]
    return columns
