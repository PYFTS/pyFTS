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
#from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import pandas as pd
from copy import deepcopy
from pyFTS.common import Util


def extract_measure(dataframe,measure,data_columns):
    if not dataframe.empty:
        df = dataframe[(dataframe.Measure == measure)][data_columns]
        tmp = df.to_dict(orient="records")[0]
        ret = [k for k in tmp.values() if not np.isnan(k)]
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


def point_dataframe_synthetic_columns():
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


def save_dataframe_point(experiments, file, objs, rmse, save, synthetic, smape, times, u):
    """
    Create a dataframe to store the benchmark results
    :param experiments: dictionary with the execution results
    :param file: 
    :param objs: 
    :param rmse: 
    :param save: 
    :param synthetic: 
    :param smape: 
    :param times: 
    :param u: 
    :return: 
    """
    ret = []

    if synthetic:

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

        columns = point_dataframe_synthetic_columns()
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
    try:
        dat = pd.DataFrame(ret, columns=columns)
        if save: dat.to_csv(Util.uniquefilename(file), sep=";", index=False)
        return dat
    except Exception as ex:
        print(ex)
        print(experiments)
        print(columns)
        print(ret)


def cast_dataframe_to_synthetic_point(infile, outfile, experiments):
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

    dat = pd.DataFrame(ret, columns=point_dataframe_synthetic_columns())
    dat.to_csv(outfile, sep=";", index=False)


def analytical_data_columns(experiments):
    data_columns = [str(k) for k in np.arange(0, experiments)]
    return data_columns


def plot_dataframe_point(file_synthetic, file_analytic, experiments, tam, save=False, file=None,
                         sort_columns=['UAVG', 'RMSEAVG', 'USTD', 'RMSESTD'],
                         sort_ascend=[1, 1, 1, 1],save_best=False,
                         ignore=None,replace=None):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=tam)

    axes[0].set_title('RMSE')
    axes[1].set_title('SMAPE')
    axes[2].set_title('U Statistic')

    dat_syn = pd.read_csv(file_synthetic, sep=";", usecols=point_dataframe_synthetic_columns())

    bests = find_best(dat_syn, sort_columns, sort_ascend)

    dat_ana = pd.read_csv(file_analytic, sep=";", usecols=point_dataframe_analytic_columns(experiments))

    data_columns = analytical_data_columns(experiments)

    if save_best:
        dat = pd.DataFrame.from_dict(bests, orient='index')
        dat.to_csv(Util.uniquefilename(file_synthetic.replace("synthetic","best")), sep=";", index=False)

    rmse = []
    smape = []
    u = []
    times = []
    labels = []

    for b in sorted(bests.keys()):
        if check_ignore_list(b, ignore):
            continue

        best = bests[b]
        tmp = dat_ana[(dat_ana.Model == best["Model"]) & (dat_ana.Order == best["Order"])
                & (dat_ana.Scheme == best["Scheme"]) & (dat_ana.Partitions == best["Partitions"])]
        rmse.append( extract_measure(tmp,'RMSE',data_columns) )
        smape.append(extract_measure(tmp, 'SMAPE', data_columns))
        u.append(extract_measure(tmp, 'U', data_columns))
        times.append(extract_measure(tmp, 'TIME', data_columns))

        labels.append(check_replace_list(best["Model"] + " " + str(best["Order"]),replace))

    axes[0].boxplot(rmse, labels=labels, autorange=True, showmeans=True)
    axes[0].set_title("RMSE")
    axes[1].boxplot(smape, labels=labels, autorange=True, showmeans=True)
    axes[1].set_title("SMAPE")
    axes[2].boxplot(u, labels=labels, autorange=True, showmeans=True)
    axes[2].set_title("U Statistic")

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)


def check_replace_list(m, replace):
    if replace is not None:
        for r in replace:
            if r[0] in m:
                return r[1]
    return m


def check_ignore_list(b, ignore):
    flag = False
    if ignore is not None:
        for i in ignore:
            if i in b:
                flag = True
    return flag


def save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, synthetic, times, q05, q25, q75, q95):
    ret = []
    if synthetic:
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
            mod.append(round(np.nanmean(q05[k]), 2))
            mod.append(round(np.nanstd(q05[k]), 2))
            mod.append(round(np.nanmean(q25[k]), 2))
            mod.append(round(np.nanstd(q25[k]), 2))
            mod.append(round(np.nanmean(q75[k]), 2))
            mod.append(round(np.nanstd(q75[k]), 2))
            mod.append(round(np.nanmean(q95[k]), 2))
            mod.append(round(np.nanstd(q95[k]), 2))
            mod.append(l)
            ret.append(mod)

        columns = interval_dataframe_synthetic_columns()
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
                tmp = [n, o, s, p, l, 'Q05']
                tmp.extend(q05[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'Q25']
                tmp.extend(q25[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'Q75']
                tmp.extend(q75[k])
                ret.append(deepcopy(tmp))
                tmp = [n, o, s, p, l, 'Q95']
                tmp.extend(q95[k])
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


def interval_dataframe_synthetic_columns():
    columns = ["Model", "Order", "Scheme", "Partitions", "SHARPAVG", "SHARPSTD", "RESAVG", "RESSTD", "COVAVG",
               "COVSTD", "TIMEAVG", "TIMESTD", "Q05AVG", "Q05STD", "Q25AVG", "Q25STD", "Q75AVG", "Q75STD", "Q95AVG", "Q95STD"]
    return columns


def cast_dataframe_to_synthetic_interval(infile, outfile, experiments):
    columns = interval_dataframe_analytic_columns(experiments)
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
                        sharpness = extract_measure(df, 'Sharpness', data_columns)
                        resolution = extract_measure(df, 'Resolution', data_columns)
                        coverage = extract_measure(df, 'Coverage', data_columns)
                        times = extract_measure(df, 'TIME', data_columns)
                        q05 = extract_measure(df, 'Q05', data_columns)
                        q25 = extract_measure(df, 'Q25', data_columns)
                        q75 = extract_measure(df, 'Q75', data_columns)
                        q95 = extract_measure(df, 'Q95', data_columns)
                        mod.append(m)
                        mod.append(o)
                        mod.append(s)
                        mod.append(p)
                        mod.append(np.round(np.nanmean(sharpness), 2))
                        mod.append(np.round(np.nanstd(sharpness), 2))
                        mod.append(np.round(np.nanmean(resolution), 2))
                        mod.append(np.round(np.nanstd(resolution), 2))
                        mod.append(np.round(np.nanmean(coverage), 2))
                        mod.append(np.round(np.nanstd(coverage), 2))
                        mod.append(np.round(np.nanmean(times), 4))
                        mod.append(np.round(np.nanstd(times), 4))
                        mod.append(np.round(np.nanmean(q05), 4))
                        mod.append(np.round(np.nanstd(q05), 4))
                        mod.append(np.round(np.nanmean(q25), 4))
                        mod.append(np.round(np.nanstd(q25), 4))
                        mod.append(np.round(np.nanmean(q75), 4))
                        mod.append(np.round(np.nanstd(q75), 4))
                        mod.append(np.round(np.nanmean(q95), 4))
                        mod.append(np.round(np.nanstd(q95), 4))
                        ret.append(mod)

    dat = pd.DataFrame(ret, columns=interval_dataframe_synthetic_columns())
    dat.to_csv(outfile, sep=";", index=False)


def plot_dataframe_interval(file_synthetic, file_analytic, experiments, tam, save=False, file=None,
                            sort_columns=['COVAVG', 'SHARPAVG', 'COVSTD', 'SHARPSTD'],
                            sort_ascend=[True, False, True, True],save_best=False,
                            ignore=None, replace=None):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=tam)

    axes[0].set_title('Sharpness')
    axes[1].set_title('Resolution')
    axes[2].set_title('Coverage')

    dat_syn = pd.read_csv(file_synthetic, sep=";", usecols=interval_dataframe_synthetic_columns())

    bests = find_best(dat_syn, sort_columns, sort_ascend)

    dat_ana = pd.read_csv(file_analytic, sep=";", usecols=interval_dataframe_analytic_columns(experiments))

    data_columns = analytical_data_columns(experiments)

    if save_best:
        dat = pd.DataFrame.from_dict(bests, orient='index')
        dat.to_csv(Util.uniquefilename(file_synthetic.replace("synthetic","best")), sep=";", index=False)

    sharpness = []
    resolution = []
    coverage = []
    times = []
    labels = []
    bounds_shp = []

    for b in sorted(bests.keys()):
        if check_ignore_list(b, ignore):
            continue
        best = bests[b]
        df = dat_ana[(dat_ana.Model == best["Model"]) & (dat_ana.Order == best["Order"])
                & (dat_ana.Scheme == best["Scheme"]) & (dat_ana.Partitions == best["Partitions"])]
        sharpness.append( extract_measure(df,'Sharpness',data_columns) )
        resolution.append(extract_measure(df, 'Resolution', data_columns))
        coverage.append(extract_measure(df, 'Coverage', data_columns))
        times.append(extract_measure(df, 'TIME', data_columns))
        labels.append(check_replace_list(best["Model"] + " " + str(best["Order"]), replace))

    axes[0].boxplot(sharpness, labels=labels, autorange=True, showmeans=True)
    axes[0].set_title("Sharpness")
    axes[1].boxplot(resolution, labels=labels, autorange=True, showmeans=True)
    axes[1].set_title("Resolution")
    axes[2].boxplot(coverage, labels=labels, autorange=True, showmeans=True)
    axes[2].set_title("Coverage")
    axes[2].set_ylim([0, 1.1])

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)



def plot_dataframe_interval_pinball(file_synthetic, file_analytic, experiments, tam, save=False, file=None,
                                    sort_columns=['COVAVG','SHARPAVG','COVSTD','SHARPSTD'],
                                    sort_ascend=[True, False, True, True], save_best=False,
                                    ignore=None, replace=None):

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=tam)
    axes[0].set_title(r'$\tau=0.05$')
    axes[1].set_title(r'$\tau=0.25$')
    axes[2].set_title(r'$\tau=0.75$')
    axes[3].set_title(r'$\tau=0.95$')

    dat_syn = pd.read_csv(file_synthetic, sep=";", usecols=interval_dataframe_synthetic_columns())

    bests = find_best(dat_syn, sort_columns, sort_ascend)

    dat_ana = pd.read_csv(file_analytic, sep=";", usecols=interval_dataframe_analytic_columns(experiments))

    data_columns = analytical_data_columns(experiments)

    if save_best:
        dat = pd.DataFrame.from_dict(bests, orient='index')
        dat.to_csv(Util.uniquefilename(file_synthetic.replace("synthetic","best")), sep=";", index=False)

    q05 = []
    q25 = []
    q75 = []
    q95 = []
    labels = []

    for b in sorted(bests.keys()):
        if check_ignore_list(b, ignore):
            continue
        best = bests[b]
        df = dat_ana[(dat_ana.Model == best["Model"]) & (dat_ana.Order == best["Order"])
                & (dat_ana.Scheme == best["Scheme"]) & (dat_ana.Partitions == best["Partitions"])]
        q05.append(extract_measure(df, 'Q05', data_columns))
        q25.append(extract_measure(df, 'Q25', data_columns))
        q75.append(extract_measure(df, 'Q75', data_columns))
        q95.append(extract_measure(df, 'Q95', data_columns))
        labels.append(check_replace_list(best["Model"] + " " + str(best["Order"]), replace))

    axes[0].boxplot(q05, labels=labels, vert=False, autorange=True, showmeans=True)
    axes[1].boxplot(q25, labels=labels, vert=False, autorange=True, showmeans=True)
    axes[2].boxplot(q75, labels=labels, vert=False, autorange=True, showmeans=True)
    axes[3].boxplot(q95, labels=labels, vert=False, autorange=True, showmeans=True)

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)


def save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, synthetic):
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
    :param synthetic: 
    :return: 
    """
    ret = []

    if synthetic:

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

        columns = ahead_dataframe_synthetic_columns()
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


def ahead_dataframe_synthetic_columns():
    columns = ["Model", "Order", "Scheme", "Partitions", "CRPS1AVG", "CRPS1STD", "CRPS2AVG", "CRPS2STD",
               "TIME1AVG", "TIME1STD", "TIME2AVG", "TIME2STD"]
    return columns


def cast_dataframe_to_synthetic_ahead(infile, outfile, experiments):
    columns = ahead_dataframe_analytic_columns(experiments)
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
                        crps1 = extract_measure(df, 'CRPS_Interval', data_columns)
                        crps2 = extract_measure(df, 'CRPS_Distribution', data_columns)
                        times1 = extract_measure(df, 'TIME_Interval', data_columns)
                        times2 = extract_measure(df, 'TIME_Distribution', data_columns)
                        mod.append(m)
                        mod.append(o)
                        mod.append(s)
                        mod.append(p)
                        mod.append(np.round(np.nanmean(crps1), 2))
                        mod.append(np.round(np.nanstd(crps1), 2))
                        mod.append(np.round(np.nanmean(crps2), 2))
                        mod.append(np.round(np.nanstd(crps2), 2))
                        mod.append(np.round(np.nanmean(times1), 2))
                        mod.append(np.round(np.nanstd(times1), 2))
                        mod.append(np.round(np.nanmean(times2), 4))
                        mod.append(np.round(np.nanstd(times2), 4))
                        ret.append(mod)

    dat = pd.DataFrame(ret, columns=ahead_dataframe_synthetic_columns())
    dat.to_csv(outfile, sep=";", index=False)


def plot_dataframe_ahead(file_synthetic, file_analytic, experiments, tam, save=False, file=None,
                         sort_columns=['CRPS1AVG', 'CRPS2AVG', 'CRPS1STD', 'CRPS2STD'],
                         sort_ascend=[True, True, True, True],save_best=False,
                         ignore=None, replace=None):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=tam)

    axes[0].set_title('CRPS Interval Ahead')
    axes[1].set_title('CRPS Distribution Ahead')

    dat_syn = pd.read_csv(file_synthetic, sep=";", usecols=ahead_dataframe_synthetic_columns())

    bests = find_best(dat_syn, sort_columns, sort_ascend)

    dat_ana = pd.read_csv(file_analytic, sep=";", usecols=ahead_dataframe_analytic_columns(experiments))

    data_columns = analytical_data_columns(experiments)

    if save_best:
        dat = pd.DataFrame.from_dict(bests, orient='index')
        dat.to_csv(Util.uniquefilename(file_synthetic.replace("synthetic","best")), sep=";", index=False)

    crps1 = []
    crps2 = []
    labels = []

    for b in sorted(bests.keys()):
        if check_ignore_list(b, ignore):
            continue
        best = bests[b]
        df = dat_ana[(dat_ana.Model == best["Model"]) & (dat_ana.Order == best["Order"])
                & (dat_ana.Scheme == best["Scheme"]) & (dat_ana.Partitions == best["Partitions"])]
        crps1.append( extract_measure(df,'CRPS_Interval',data_columns) )
        crps2.append(extract_measure(df, 'CRPS_Distribution', data_columns))
        labels.append(check_replace_list(best["Model"] + " " + str(best["Order"]), replace))

    axes[0].boxplot(crps1, labels=labels, autorange=True, showmeans=True)
    axes[1].boxplot(crps2, labels=labels, autorange=True, showmeans=True)

    plt.tight_layout()
    Util.showAndSaveImage(fig, file, save)

