#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.cross_validation import KFold
from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid
from pyFTS.common import Membership, FuzzySet, FLR, Transformations
import time

current_milli_time = lambda: int(round(time.time() * 1000))

def getIntervalStatistics(original, models):
    ret = "Model		& RMSE		& MAPE		& Sharpness		& Resolution		& Coverage	\\ \n"
    for fts in models:
        forecasts = fts.forecast(original)
        ret = ret + fts.shortname + "		& "
        ret = ret + str(round(Measures.rmse_interval(original[fts.order - 1:], forecasts), 2)) + "		& "
        ret = ret + str(round(Measures.mape_interval(original[fts.order - 1:], forecasts), 2)) + "		& "
        ret = ret + str(round(Measures.sharpness(forecasts), 2)) + "		& "
        ret = ret + str(round(Measures.resolution(forecasts), 2)) + "		& "
        ret = ret + str(round(Measures.coverage(original[fts.order - 1:], forecasts), 2)) + "	\\ \n"
    return ret


def plotDistribution(dist):
    for k in dist.index:
        alpha = np.array([dist[x][k] for x in dist]) * 100
        x = [k for x in np.arange(0, len(alpha))]
        y = dist.columns
        plt.scatter(x, y, c=alpha, marker='s', linewidths=0, cmap='Oranges', norm=pltcolors.Normalize(vmin=0, vmax=1),
                    vmin=0, vmax=1, edgecolors=None)


def uniquefilename(name):
    if '.' in name:
        tmp = name.split('.')
        return  tmp[0] + str(current_milli_time()) + '.' + tmp[1]
    else:
        return name + str(current_milli_time())



def plotComparedSeries(original, models, colors, typeonlegend=False, save=False, file=None,tam=[20, 5]):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    mi = []
    ma = []

    ax.plot(original, color='black', label="Original")
    count = 0
    for fts in models:
        if fts.hasPointForecasting:
            forecasted = fts.forecast(original)
            mi.append(min(forecasted))
            ma.append(max(forecasted))
            for k in np.arange(0, fts.order):
                forecasted.insert(0, None)
            lbl = fts.shortname
            if typeonlegend: lbl += " (Point)"
            ax.plot(forecasted, color=colors[count], label=lbl, ls="-")

        if fts.hasIntervalForecasting:
            forecasted = fts.forecastInterval(original)
            lower = [kk[0] for kk in forecasted]
            upper = [kk[1] for kk in forecasted]
            mi.append(min(lower))
            ma.append(max(upper))
            for k in np.arange(0, fts.order):
                lower.insert(0, None)
                upper.insert(0, None)
            lbl = fts.shortname
            if typeonlegend: lbl += " (Interval)"
            ax.plot(lower, color=colors[count], label=lbl,ls="--")
            ax.plot(upper, color=colors[count],ls="--")

        handles0, labels0 = ax.get_legend_handles_labels()
        ax.legend(handles0, labels0, loc=2)
        count = count + 1
    # ax.set_title(fts.name)
    ax.set_ylim([min(mi), max(ma)])
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])

    if save:
        plt.show()
        fig.savefig(uniquefilename(file))
        plt.close(fig)


def plotComparedIntervalsAhead(original, models, colors, distributions, time_from, time_to,
                               interpol=False, save=False, file=None,tam=[20, 5], resolution=None):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    if resolution is None: resolution = (max(original) - min(original))/100

    mi = []
    ma = []

    count = 0
    for fts in models:
        if fts.hasDistributionForecasting and distributions[count]:
            density = fts.forecastAheadDistribution(original[time_from - fts.order:time_from], time_to, resolution)

            y = density.columns
            t = len(y)

            # interpol between time_from and time_from+1
            #if interpol:
            #    diffs = [density[q][0] / 50 for q in density]
            #    for p in np.arange(0, 50):
            #        xx = [(time_from - 1) + 0.02 * p for q in np.arange(0, t)]
            #        alpha2 = np.array([diffs[q] * p for q in np.arange(0, t)]) * 100
            #        ax.scatter(xx, y, c=alpha2, marker='s', linewidths=0, cmap='Oranges',
            #                   norm=pltcolors.Normalize(vmin=0, vmax=1), vmin=0, vmax=1, edgecolors=None)
            for k in density.index:
                alpha = np.array([density[q][k] for q in density]) * 100

                x = [time_from  + k for x in np.arange(0, t)]

                for cc in np.arange(0,resolution,5):
                    ax.scatter(x, y+cc, c=alpha, marker='s', linewidths=0, cmap='Oranges',
                               norm=pltcolors.Normalize(vmin=0, vmax=1), vmin=0, vmax=1, edgecolors=None)
                if interpol and k < max(density.index):
                    diffs = [(density[q][k + 1] - density[q][k])/50 for q in density]
                    for p in np.arange(0,50):
                        xx = [time_from + k + 0.02*p for q in np.arange(0, t)]
                        alpha2 = np.array([density[density.columns[q]][k] + diffs[q]*p for q in np.arange(0, t)]) * 100
                        ax.scatter(xx, y, c=alpha2, marker='s', linewidths=0, cmap='Oranges',
                                   norm=pltcolors.Normalize(vmin=0, vmax=1), vmin=0, vmax=1, edgecolors=None)

        if fts.hasIntervalForecasting:
            forecasts = fts.forecastAheadInterval(original[time_from - fts.order:time_from], time_to)
            lower = [kk[0] for kk in forecasts]
            upper = [kk[1] for kk in forecasts]
            mi.append(min(lower))
            ma.append(max(upper))
            for k in np.arange(0, time_from-fts.order):
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

        count = count + 1
    ax.plot(original, color='black', label="Original")
    handles0, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles0, labels0, loc=2)
    # ax.set_title(fts.name)
    ax.set_ylim([min(mi), max(ma)])
    ax.set_ylabel('F(T)')
    ax.set_xlabel('T')
    ax.set_xlim([0, len(original)])

    if save:
        plt.show()
        fig.savefig(uniquefilename(file))
        plt.close(fig)


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


from pyFTS import hwang


def HOSelecaoSimples_MenorRMSE(original, parameters, orders):
    ret = []
    errors = np.array([[0 for k in range(len(parameters))] for kk in range(len(orders))])
    forecasted_best = []
    print("Série Original")
    fig = plt.figure(figsize=[20, 12])
    fig.suptitle("Comparação de modelos ")
    ax0 = fig.add_axes([0, 0.5, 0.6, 0.45])  # left, bottom, width, height
    ax0.set_xlim([0, len(original)])
    ax0.set_ylim([min(original), max(original)])
    ax0.set_title('Série Temporal')
    ax0.set_ylabel('F(T)')
    ax0.set_xlabel('T')
    ax0.plot(original, label="Original")
    min_rmse = 100000.0
    best = None
    pc = 0
    for p in parameters:
        oc = 0
        for o in orders:
            sets = Grid.GridPartitionerTrimf(original, p)
            fts = hwang.HighOrderFTS(o, "k = " + str(p) + " w = " + str(o))
            fts.train(original, sets)
            forecasted = [fts.forecast(original, xx) for xx in range(o, len(original))]
            error = Measures.rmse(np.array(forecasted), np.array(original[o:]))
            for kk in range(o):
                forecasted.insert(0, None)
            ax0.plot(forecasted, label=fts.name)
            print(o, p, error)
            errors[oc, pc] = error
            if error < min_rmse:
                min_rmse = error
                best = fts
                forecasted_best = forecasted
            oc = oc + 1
        pc = pc + 1
        handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0)
    ax1 = Axes3D(fig, rect=[0.6, 0.5, 0.45, 0.45], elev=30, azim=144)
    # ax1 = fig.add_axes([0.6, 0.5, 0.45, 0.45], projection='3d')
    ax1.set_title('Comparação dos Erros Quadráticos Médios por tamanho da janela')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Quantidade de Partições')
    ax1.set_zlabel('W')
    X, Y = np.meshgrid(parameters, orders)
    surf = ax1.plot_surface(X, Y, errors, rstride=1, cstride=1, antialiased=True)
    ret.append(best)
    ret.append(forecasted_best)

    # Modelo diferencial
    print("\nSérie Diferencial")
    errors = np.array([[0 for k in range(len(parameters))] for kk in range(len(orders))])
    forecastedd_best = []
    ax2 = fig.add_axes([0, 0, 0.6, 0.45])  # left, bottom, width, height
    ax2.set_xlim([0, len(original)])
    ax2.set_ylim([min(original), max(original)])
    ax2.set_title('Série Temporal')
    ax2.set_ylabel('F(T)')
    ax2.set_xlabel('T')
    ax2.plot(original, label="Original")
    min_rmse = 100000.0
    bestd = None
    pc = 0
    for p in parameters:
        oc = 0
        for o in orders:
            sets = Grid.GridPartitionerTrimf(Transformations.differential(original), p)
            fts = hwang.HighOrderFTS(o, "k = " + str(p) + " w = " + str(o))
            fts.train(original, sets)
            forecasted = [fts.forecastDiff(original, xx) for xx in range(o, len(original))]
            error = Measures.rmse(np.array(forecasted), np.array(original[o:]))
            for kk in range(o):
                forecasted.insert(0, None)
            ax2.plot(forecasted, label=fts.name)
            print(o, p, error)
            errors[oc, pc] = error
            if error < min_rmse:
                min_rmse = error
                bestd = fts
                forecastedd_best = forecasted
            oc = oc + 1
        pc = pc + 1
    handles0, labels0 = ax2.get_legend_handles_labels()
    ax2.legend(handles0, labels0)
    ax3 = Axes3D(fig, rect=[0.6, 0.0, 0.45, 0.45], elev=30, azim=144)
    # ax3 = fig.add_axes([0.6, 0.0, 0.45, 0.45], projection='3d')
    ax3.set_title('Comparação dos Erros Quadráticos Médios')
    ax3.set_ylabel('RMSE')
    ax3.set_xlabel('Quantidade de Partições')
    ax3.set_zlabel('W')
    X, Y = np.meshgrid(parameters, orders)
    surf = ax3.plot_surface(X, Y, errors, rstride=1, cstride=1, antialiased=True)
    ret.append(bestd)
    ret.append(forecastedd_best)
    return ret
