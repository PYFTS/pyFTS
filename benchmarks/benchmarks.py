#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cross_validation import KFold
from pyFTS.benchmarks import Measures, naive, ResidualAnalysis
from pyFTS.partitioners import Grid
from pyFTS.common import Membership, FuzzySet, FLR, Transformations, Util
from pyFTS import fts, chen, yu, ismailefendi, sadaei, hofts, hwang, pfts, ifts

colors = ['grey', 'rosybrown', 'maroon', 'red','orange', 'yellow', 'olive', 'green',
          'cyan', 'blue', 'darkblue', 'purple', 'darkviolet']

ncol = len(colors)

styles = ['-','--','-.',':','.']

nsty = len(styles)

def allPointForecasters(data_train, data_test, partitions, max_order=3, statistics=True, residuals=True,
                        series=True, save=False, file=None, tam=[20, 5], models=None, transformation=None):

    if models is None:
        models = [naive.Naive, chen.ConventionalFTS, yu.WeightedFTS, ismailefendi.ImprovedWeightedFTS,
                  sadaei.ExponentialyWeightedFTS, hofts.HighOrderFTS,  pfts.ProbabilisticFTS]

    objs = []

    if transformation is not None:
        data_train_fs = Grid.GridPartitionerTrimf(transformation.apply(data_train),partitions)
    else:
        data_train_fs = Grid.GridPartitionerTrimf(data_train, partitions)

    count = 1

    lcolors = []

    for count, model in enumerate(models, start=0):
        #print(model)
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

    if statistics:
        print(getPointStatistics(data_test, objs))

    if residuals:
        print(ResidualAnalysis.compareResiduals(data_test, objs))
        ResidualAnalysis.plotResiduals2(data_test, objs, save=save, file=file, tam=tam)

    if series:
        plotComparedSeries(data_test, objs, lcolors, typeonlegend=False, save=save, file=file, tam=tam,
                           intervals=False)


def getPointStatistics(data, models, externalmodels = None, externalforecasts = None, indexers=None):
    ret = "Model		& Order     & RMSE		& SMAPE      & Theil's U		\\\\ \n"
    for count,model in enumerate(models,start=0):
        if indexers is not None and indexers[count] is not None:
            ndata = np.array(indexers[count].get_data(data[model.order:]))
        else:
            ndata = np.array(data[model.order:])

        if model.isMultivariate or indexers is None:
            forecasts = model.forecast(data)
        elif not model.isMultivariate and indexers is not None:
            forecasts = model.forecast( indexers[count].get_data(data) )

        if model.hasSeasonality:
            nforecasts = np.array(forecasts)
        else:
            nforecasts = np.array(forecasts[:-1])
        ret += model.shortname + "		& "
        ret += str(model.order) + "		& "
        ret += str(round(Measures.rmse(ndata, nforecasts), 2)) + "		& "
        ret += str(round(Measures.smape(ndata, nforecasts), 2))+ "		& "
        ret += str(round(Measures.UStatistic(ndata, nforecasts), 2))
        #ret += str(round(Measures.TheilsInequality(np.array(data[fts.order:]), np.array(forecasts[:-1])), 4))
        ret += "	\\\\ \n"
    if externalmodels is not None:
        l = len(externalmodels)
        for k in np.arange(0,l):
            ret += externalmodels[k] + "		& "
            ret += " 1		& "
            ret += str(round(Measures.rmse(data, externalforecasts[k][:-1]), 2)) + "		& "
            ret += str(round(Measures.smape(data, externalforecasts[k][:-1]), 2))+ "		& "
            ret += str(round(Measures.UStatistic(np.array(data), np.array(forecasts[:-1])), 2))
            ret += "	\\\\ \n"
    return ret


def allIntervalForecasters(data_train, data_test, partitions, max_order=3,save=False, file=None, tam=[20, 5],
                           models=None, transformation=None):
    if models is None:
        models = [ifts.IntervalFTS, pfts.ProbabilisticFTS]

    objs = []

    if transformation is not None:
        data_train_fs = Grid.GridPartitionerTrimf(transformation.apply(data_train),partitions)
    else:
        data_train_fs = Grid.GridPartitionerTrimf(data_train, partitions)

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

    print(getIntervalStatistics(data_test, objs))

    plotComparedSeries(data_test, objs, lcolors, typeonlegend=False, save=save, file=file, tam=tam,  intervals=True)


def getIntervalStatistics(original, models):
    ret = "Model	& Order     & Sharpness		& Resolution		& Coverage	\\\\ \n"
    for fts in models:
        forecasts = fts.forecastInterval(original)
        ret += fts.shortname + "		& "
        ret += str(fts.order) + "		& "
        ret += str(round(Measures.sharpness(forecasts), 2)) + "		& "
        ret += str(round(Measures.resolution(forecasts), 2)) + "		& "
        ret += str(round(Measures.coverage(original[fts.order:], forecasts[:-1]), 2)) + "	\\\\ \n"
    return ret


def plotDistribution(dist):
    for k in dist.index:
        alpha = np.array([dist[x][k] for x in dist]) * 100
        x = [k for x in np.arange(0, len(alpha))]
        y = dist.columns
        plt.scatter(x, y, c=alpha, marker='s', linewidths=0, cmap='Oranges', norm=pltcolors.Normalize(vmin=0, vmax=1),
                    vmin=0, vmax=1, edgecolors=None)


def plotComparedSeries(original, models, colors, typeonlegend=False, save=False, file=None, tam=[20, 5],
                       intervals=True):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    mi = []
    ma = []

    legends = []

    ax.plot(original, color='black', label="Original", linewidth=1.5)

    for count, fts in enumerate(models, start=0):
        if fts.hasPointForecasting and not intervals:
            forecasted = fts.forecast(original)
            mi.append(min(forecasted) * 0.95)
            ma.append(max(forecasted) * 1.05)
            for k in np.arange(0, fts.order):
                forecasted.insert(0, None)
            lbl = fts.shortname
            if typeonlegend: lbl += " (Point)"
            ax.plot(forecasted, color=colors[count], label=lbl, ls="-")

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
            ax.plot(lower, color=colors[count], label=lbl, ls="-")
            ax.plot(upper, color=colors[count], ls="-")

        handles0, labels0 = ax.get_legend_handles_labels()
        lgd = ax.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))
        legends.append(lgd)

    # ax.set_title(fts.name)
    ax.set_ylim([min(mi), max(ma)])
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])

    Util.showAndSaveImage(fig, file, save, lgd=legends)

def allAheadForecasters(data_train, data_test, partitions, start, steps, resolution = None, max_order=3,save=False, file=None, tam=[20, 5],
                           models=None, transformation=None, option=2):
    if models is None:
        models = [pfts.ProbabilisticFTS]

    if resolution is None: resolution = (max(data_train) - min(data_train)) / 100

    objs = []

    if transformation is not None:
        data_train_fs = Grid.GridPartitionerTrimf(transformation.apply(data_train),partitions)
    else:
        data_train_fs = Grid.GridPartitionerTrimf(data_train, partitions)

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

    print(getDistributionStatistics(data_test[start:], objs, steps, resolution))

    #plotComparedIntervalsAhead(data_test, objs, lcolors, distributions=, save=save, file=file, tam=tam,  intervals=True)


def getDistributionStatistics(original, models, steps, resolution):
    ret = "Model	& Order     &  Interval & Distribution	\\\\ \n"
    for fts in models:
        densities1 = fts.forecastAheadDistribution(original,steps,resolution, parameters=3)
        densities2 = fts.forecastAheadDistribution(original, steps, resolution, parameters=2)
        ret += fts.shortname + "		& "
        ret += str(fts.order) + "		& "
        ret += str(round(Measures.crps(original, densities1), 3)) + "		& "
        ret += str(round(Measures.crps(original, densities2), 3)) + "	\\\\ \n"
    return ret


def plotComparedIntervalsAhead(original, models, colors, distributions, time_from, time_to,
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
            density = fts.forecastAheadDistribution(original[time_from - fts.order:time_from],
                                                    time_to, resolution, parameters=option)

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


def SelecaoKFold_MenorRMSE(original, parameters, modelo):
    nfolds = 5
    ret = []
    errors = np.array([[0 for k in parameters] for z in np.arange(0, nfolds)])
    forecasted_best = []
    print("Série Original")
    fig = plt.figure(figsize=[18, 10])
    fig.suptitle("Comparação de modelos ")
    ax0 = fig.add_axes([0, 0.5, 0.65, 0.45])  # left, bottom, width, height
    ax0.set_xlim([0, len(original)])
    ax0.set_ylim([min(original), max(original)])
    ax0.set_title('Série Temporal')
    ax0.set_ylabel('F(T)')
    ax0.set_xlabel('T')
    ax0.plot(original, label="Original")
    min_rmse_fold = 100000.0
    best = None
    fc = 0  # Fold count
    kf = KFold(len(original), n_folds=nfolds)
    for train_ix, test_ix in kf:
        train = original[train_ix]
        test = original[test_ix]
        min_rmse = 100000.0
        best_fold = None
        forecasted_best_fold = []
        errors_fold = []
        pc = 0  # Parameter count
        for p in parameters:
            sets = Grid.GridPartitionerTrimf(train, p)
            fts = modelo(str(p) + " particoes")
            fts.train(train, sets)
            forecasted = [fts.forecast(xx) for xx in test]
            error = Measures.rmse(np.array(forecasted), np.array(test))
            errors_fold.append(error)
            print(fc, p, error)
            errors[fc, pc] = error
            if error < min_rmse:
                min_rmse = error
                best_fold = fts
                forecasted_best_fold = forecasted
            pc = pc + 1
        forecasted_best_fold = [best_fold.forecast(xx) for xx in original]
        ax0.plot(forecasted_best_fold, label=best_fold.name)
        if np.mean(errors_fold) < min_rmse_fold:
            min_rmse_fold = np.mean(errors)
            best = best_fold
            forecasted_best = forecasted_best_fold
        fc = fc + 1
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0)
    ax1 = Axes3D(fig, rect=[0.7, 0.5, 0.3, 0.45], elev=30, azim=144)
    # ax1 = fig.add_axes([0.6, 0.0, 0.45, 0.45], projection='3d')
    ax1.set_title('Comparação dos Erros Quadráticos Médios')
    ax1.set_zlabel('RMSE')
    ax1.set_xlabel('K-fold')
    ax1.set_ylabel('Partições')
    X, Y = np.meshgrid(np.arange(0, nfolds), parameters)
    surf = ax1.plot_surface(X, Y, errors.T, rstride=1, cstride=1, antialiased=True)
    ret.append(best)
    ret.append(forecasted_best)

    # Modelo diferencial
    print("\nSérie Diferencial")
    errors = np.array([[0 for k in parameters] for z in np.arange(0, nfolds)])
    forecastedd_best = []
    ax2 = fig.add_axes([0, 0, 0.65, 0.45])  # left, bottom, width, height
    ax2.set_xlim([0, len(original)])
    ax2.set_ylim([min(original), max(original)])
    ax2.set_title('Série Temporal')
    ax2.set_ylabel('F(T)')
    ax2.set_xlabel('T')
    ax2.plot(original, label="Original")
    min_rmse = 100000.0
    min_rmse_fold = 100000.0
    bestd = None
    fc = 0
    diff = Transformations.differential(original)
    kf = KFold(len(original), n_folds=nfolds)
    for train_ix, test_ix in kf:
        train = diff[train_ix]
        test = diff[test_ix]
        min_rmse = 100000.0
        best_fold = None
        forecasted_best_fold = []
        errors_fold = []
        pc = 0
        for p in parameters:
            sets = Grid.GridPartitionerTrimf(train, p)
            fts = modelo(str(p) + " particoes")
            fts.train(train, sets)
            forecasted = [fts.forecastDiff(test, xx) for xx in np.arange(len(test))]
            error = Measures.rmse(np.array(forecasted), np.array(test))
            print(fc, p, error)
            errors[fc, pc] = error
            errors_fold.append(error)
            if error < min_rmse:
                min_rmse = error
                best_fold = fts
            pc = pc + 1
        forecasted_best_fold = [best_fold.forecastDiff(original, xx) for xx in np.arange(len(original))]
        ax2.plot(forecasted_best_fold, label=best_fold.name)
        if np.mean(errors_fold) < min_rmse_fold:
            min_rmse_fold = np.mean(errors)
            best = best_fold
            forecasted_best = forecasted_best_fold
        fc = fc + 1
    handles0, labels0 = ax2.get_legend_handles_labels()
    ax2.legend(handles0, labels0)
    ax3 = Axes3D(fig, rect=[0.7, 0, 0.3, 0.45], elev=30, azim=144)
    # ax1 = fig.add_axes([0.6, 0.0, 0.45, 0.45], projection='3d')
    ax3.set_title('Comparação dos Erros Quadráticos Médios')
    ax3.set_zlabel('RMSE')
    ax3.set_xlabel('K-fold')
    ax3.set_ylabel('Partições')
    X, Y = np.meshgrid(np.arange(0, nfolds), parameters)
    surf = ax3.plot_surface(X, Y, errors.T, rstride=1, cstride=1, antialiased=True)
    ret.append(best)
    ret.append(forecasted_best)
    return ret


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
                      plotforecasts=False, elev=30, azim=144, intervals=False,parameters=None):
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

        sets = Grid.GridPartitionerTrimf(train, p)
        for oc, o in enumerate(orders, start=0):
            fts = model("q = " + str(p) + " n = " + str(o))
            fts.train(train, sets, o,parameters=parameters)
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
        fts = pfts.ProbabilisticFTS("")
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
        fts = pfts.ProbabilisticFTS("")
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
