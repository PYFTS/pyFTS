"""
Facility methods for pyFTS partitioners module
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from pyFTS.benchmarks import Measures
from pyFTS.common import Membership, Util
from pyFTS.partitioners import Grid,Huarng,FCM,Entropy

all_methods = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner, Huarng.HuarngPartitioner]

mfs = [Membership.trimf, Membership.gaussmf, Membership.trapmf]


def sliding_window_simple_search(data, windowsize, model, partitions, orders, **kwargs):

    _3d = len(orders) > 1
    ret = []
    errors = np.array([[0 for k in range(len(partitions))] for kk in range(len(orders))])
    forecasted_best = []

    figsize = kwargs.get('figsize', [10, 15])
    fig = plt.figure(figsize=figsize)

    plotforecasts = kwargs.get('plotforecasts',False)
    if plotforecasts:
        ax0 = fig.add_axes([0, 0.4, 0.9, 0.5])  # left, bottom, width, height
        ax0.set_xlim([0, len(data)])
        ax0.set_ylim([min(data) * 0.9, max(data) * 1.1])
        ax0.set_title('Forecasts')
        ax0.set_ylabel('F(T)')
        ax0.set_xlabel('T')
    min_rmse = 1000000.0
    best = None

    intervals = kwargs.get('intervals',False)
    threshold = kwargs.get('threshold',0.5)

    progressbar = kwargs.get('progressbar', None)

    rng1 = enumerate(partitions, start=0)

    if progressbar:
        from tqdm import tqdm
        rng1 = enumerate(tqdm(partitions), start=0)

    for pc, p in rng1:
        fs = Grid.GridPartitioner(data=data, npart=p)

        rng2 = enumerate(orders, start=0)

        if progressbar:
            rng2 = enumerate(tqdm(orders), start=0)

        for oc, o in rng2:
            _error = []
            for ct, train, test in Util.sliding_window(data, windowsize, 0.8, **kwargs):
                fts = model("q = " + str(p) + " n = " + str(o), partitioner=fs)
                fts.fit(train, order=o)
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
            if (min_rmse - error) > threshold:
                min_rmse = error
                best = fts
                forecasted_best = forecasted

    # print(min_rmse)
    if plotforecasts:
        # handles0, labels0 = ax0.get_legend_handles_labels()
        # ax0.legend(handles0, labels0)
        elev = kwargs.get('elev', 30)
        azim = kwargs.get('azim', 144)
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

    file = kwargs.get('file', None)
    save = kwargs.get('save', False)

    Util.show_and_save_image(fig, file, save)

    return ret


def plot_sets(data, sets, titles, tam=[12, 10], save=False, file=None):
    num = len(sets)
    #fig = plt.figure(figsize=tam)
    maxx = max(data)
    minx = min(data)
    #h = 1/num
    #print(h)
    fig, axes = plt.subplots(nrows=num, ncols=1,figsize=tam)
    for k in np.arange(0,num):
        ticks = []
        x = []
        ax = axes[k]
        ax.set_title(titles[k])
        ax.set_ylim([0, 1.1])
        for key in sets[k].keys():
            s = sets[k][key]
            if s.mf == Membership.trimf:
                ax.plot(s.parameters,[0,1,0])
            elif s.mf == Membership.gaussmf:
                tmpx = [ kk for kk in np.arange(s.lower, s.upper)]
                tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
                ax.plot(tmpx, tmpy)
            elif s.mf == Membership.trapmf:
                ax.plot(s.parameters, [0, 1, 1, 0])
            ticks.append(str(round(s.centroid, 0)) + '\n' + s.name)
            x.append(s.centroid)
        ax.xaxis.set_ticklabels(ticks)
        ax.xaxis.set_ticks(x)

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)


def plot_partitioners(data, objs, tam=[12, 10], save=False, file=None):
    sets = [k.sets for k in objs]
    titles = [k.name for k in objs]
    plot_sets(data, sets, titles, tam, save, file)


def explore_partitioners(data, npart, methods=None, mf=None, tam=[12, 10], save=False, file=None):
    if methods is None:
        methods = all_methods

    if mf is None:
        mf = mfs

    objs = []

    for p in methods:
        for m in mf:
            obj = p(data=data, npart=npart, func=m)
            obj.name = obj.name  + " - " + obj.membership_function.__name__
            objs.append(obj)

    plot_partitioners(data, objs, tam, save, file)

    return objs
