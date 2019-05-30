"""
Common facilities for pyFTS
"""

import time
import matplotlib.pyplot as plt
import dill
import numpy as np
import pandas as pd
import matplotlib.cm as cmx
import matplotlib.colors as pltcolors
from pyFTS.probabilistic import ProbabilityDistribution
from pyFTS.common import Transformations




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

    show_and_save_image(fig, file, save, lgd=lgd)



def plot_density_rectange(ax, cmap, density, fig, resolution, time_from, time_to):
    """
    Auxiliar function to plot_compared_intervals_ahead
    """
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


def plot_probability_distributions(pmfs, lcolors, tam=[15, 7]):
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111)

    for k,m in enumerate(pmfs,start=0):
        m.plot(ax, color=lcolors[k])

    handles0, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles0, labels0)

def plot_distribution(ax, cmap, probabilitydist, fig, time_from, reference_data=None):
    '''
    Plot forecasted ProbabilityDistribution objects on a matplotlib axis

    :param ax: matplotlib axis
    :param cmap: matplotlib colormap name
    :param probabilitydist: list of ProbabilityDistribution objects
    :param fig: matplotlib figure
    :param time_from: starting time (on x axis) to begin the plots
    :param reference_data:
    :return:
    '''
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


def plot_distribution2(probabilitydist, data, **kwargs):
    '''
    Plot distributions over the time (x-axis)

    :param probabilitydist: the forecasted probability distributions to plot
    :param data: the original test sample
    :keyword start_at: the time index (inside of data) to start to plot the probability distributions
    :keyword ax: a matplotlib axis. If no value was provided a new axis is created.
    :keyword cmap: a matplotlib colormap name, the default value is 'Blues'
    :keyword quantiles: the list of quantiles intervals to plot, e. g. [.05, .25, .75, .95]
    :keyword median: a boolean value indicating if the median value will be plot.
    '''
    import matplotlib.colorbar as cbar
    import matplotlib.cm as cm

    order = kwargs.get('order', 1)

    ax = kwargs.get('ax',None)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15, 5])

    l = len(probabilitydist)

    cmap = kwargs.get('cmap','Blues')
    cmap = plt.get_cmap(cmap)

    start_at = kwargs.get('start_at',0) + order - 1

    x = [k + start_at for k in range(l + 1)]

    qt = kwargs.get('quantiles',None)

    if qt is None:
        qt = [round(k, 2) for k in np.arange(.05, 1., .05)]
        qt.insert(0, .01)
        qt.append(.99)

    lq = len(qt)

    normal = plt.Normalize(min(qt), max(qt))
    scalarMap = cm.ScalarMappable(norm=normal, cmap=cmap)

    for ct in np.arange(1, int(lq / 2) + 1):
        y = [[data[start_at], data[start_at]]]
        for pd in probabilitydist:
            qts = pd.quantile([qt[ct - 1], qt[-ct]])
            y.append(qts)

        ax.fill_between(x, [k[0] for k in y], [k[1] for k in y],
                        facecolor=scalarMap.to_rgba(ct / lq))

    if kwargs.get('median',True):
        y = [data[start_at]]
        for pd in probabilitydist:
            qts = pd.quantile(.5)
            y.append(qts[0])

        ax.plot(x, y, color='red', label='Median')

    cax, _ = cbar.make_axes(ax)
    cb = cbar.ColorbarBase(cax, cmap=cmap, norm=normal)
    cb.set_label('Density')


def plot_interval(axis, intervals, order, label, color='red', typeonlegend=False, ls='-', linewidth=1):
    '''
    Plot forecasted intervals on matplotlib

    :param axis: matplotlib axis
    :param intervals: list of forecasted intervals
    :param order: order of the model that create the forecasts
    :param label: figure label
    :param color: matplotlib color name
    :param typeonlegend:
    :param ls: matplotlib line style
    :param linewidth: matplotlib width
    :return:
    '''
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


def plot_rules(model, size=[5, 5], axis=None, rules_by_axis=None, columns=1):
    '''
    Plot the FLRG rules of a FTS model on a matplotlib axis

    :param model: FTS model
    :param size: figure size
    :param axis: matplotlib axis
    :param rules_by_axis: number of rules plotted by column
    :param columns: number of columns
    :return:
    '''
    if axis is None and rules_by_axis is None:
        rows = 1
    elif axis is None and rules_by_axis is not None:
        rows = (((len(model.flrgs.keys())//rules_by_axis)) // columns)+1

    fig, axis = plt.subplots(nrows=rows, ncols=columns, figsize=size)

    if rules_by_axis is None:
        draw_sets_on_axis(axis, model, size)

    _lhs = model.partitioner.ordered_sets if not model.is_high_order else model.flrgs.keys()

    for ct, key in enumerate(_lhs):

        xticks = []
        xtickslabels = []

        if rules_by_axis is None:
            ax = axis
        else:
            colcount = (ct // rules_by_axis) % columns
            rowcount = (ct // rules_by_axis) // columns

            if rows > 1 and columns > 1:
                ax = axis[rowcount, colcount]
            elif columns > 1:
                ax = axis[rowcount]
            else:
                ax = axis

            if ct % rules_by_axis == 0:
                draw_sets_on_axis(ax, model, size)

        if not model.is_high_order:
            if key in model.flrgs:
                x = (ct % rules_by_axis) + 1
                flrg = model.flrgs[key]
                y = model.sets[key].centroid
                ax.plot([x],[y],'o')
                xticks.append(x)
                xtickslabels.append(key)
                for rhs in flrg.RHS:
                    dest = model.sets[rhs].centroid
                    ax.arrow(x+.1, y, 0.8, dest - y, #length_includes_head=True,
                               head_width=0.1, head_length=0.1, shape='full', overhang=0,
                               fc='k', ec='k')
        else:
            flrg = model.flrgs[key]
            x = (ct%rules_by_axis)*model.order + 1
            for ct2, lhs in enumerate(flrg.LHS):
                y = model.sets[lhs].centroid
                ax.plot([x+ct2], [y], 'o')
                xticks.append(x+ct2)
                xtickslabels.append(lhs)
            for ct2 in range(1, model.order):
                fs1 = flrg.LHS[ct2-1]
                fs2 = flrg.LHS[ct2]
                y = model.sets[fs1].centroid
                dest = model.sets[fs2].centroid
                ax.plot([x+ct2-1,x+ct2], [y,dest],'-')

            y = model.sets[flrg.LHS[-1]].centroid
            for rhs in flrg.RHS:
                dest = model.sets[rhs].centroid
                ax.arrow(x + model.order -1 + .1, y, 0.8, dest - y,  # length_includes_head=True,
                           head_width=0.1, head_length=0.1, shape='full', overhang=0,
                           fc='k', ec='k')


        ax.set_xticks(xticks)
        ax.set_xticklabels(xtickslabels)
        ax.set_xlim([0,rules_by_axis*model.order+1])

    plt.tight_layout()
    plt.show()


def draw_sets_on_axis(axis, model, size):
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=size)
    for ct, key in enumerate(model.partitioner.ordered_sets):
        fs = model.sets[key]
        axis.plot([0, 1, 0], fs.parameters, label=fs.name)
        axis.axhline(fs.centroid, c="lightgray", alpha=0.5)
    axis.set_xlim([0, len(model.partitioner.ordered_sets)])
    axis.set_xticks(range(0, len(model.partitioner.ordered_sets)))
    tmp = ['']
    tmp.extend(model.partitioner.ordered_sets)
    axis.set_xticklabels(tmp)
    axis.set_ylim([model.partitioner.min, model.partitioner.max])
    axis.set_yticks([model.sets[k].centroid for k in model.partitioner.ordered_sets])
    axis.set_yticklabels([str(round(model.sets[k].centroid, 1)) + " - " + k
                          for k in model.partitioner.ordered_sets])


current_milli_time = lambda: int(round(time.time() * 1000))


def uniquefilename(name):
    if '.' in name:
        tmp = name.split('.')
        return  tmp[0] + str(current_milli_time()) + '.' + tmp[1]
    else:
        return name + str(current_milli_time())



def show_and_save_image(fig, file, flag, lgd=None):
    """
    Show and image and save on file

    :param fig: Matplotlib Figure object
    :param file: filename to save the picture
    :param flag: if True the image will be saved
    :param lgd: legend
    """
    plt.show()
    if flag:
        if lgd is not None:
            fig.savefig(file, additional_artists=lgd,bbox_inches='tight')  #bbox_extra_artists=(lgd,), )
        else:
            fig.savefig(file)
        plt.close(fig)


def enumerate2(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step


def sliding_window(data, windowsize, train=0.8, inc=0.1, **kwargs):
    """
    Sliding window method of cross validation for time series

    :param data: the entire dataset
    :param windowsize: window size
    :param train: percentual of the window size will be used for training the models
    :param inc: percentual of data used for slide the window
    :return: window count, training set, test set
    """

    multivariate = True if isinstance(data, pd.DataFrame) else False

    l = len(data) if not multivariate else len(data.index)
    ttrain = int(round(windowsize * train, 0))
    ic = int(round(windowsize * inc, 0))

    progressbar = kwargs.get('progress', None)

    rng = np.arange(0,l-windowsize+ic,ic)

    if progressbar:
        from tqdm import tqdm
        rng = tqdm(rng)

    for count in rng:
        if count + windowsize > l:
            _end = l
        else:
            _end = count + windowsize
        if multivariate:
            yield (count, data.iloc[count: count + ttrain], data.iloc[count + ttrain: _end])
        else:
            yield (count,  data[count : count + ttrain], data[count + ttrain : _end]  )


def persist_obj(obj, file):
    """
    Persist an object on filesystem. This function depends on Dill package

    :param obj: object on memory
    :param file: file name to store the object
    """
    try:
        with open(file, 'wb') as _file:
            dill.dump(obj, _file)
    except Exception as ex:
        print("File {} could not be saved due exception {}".format(file, ex))


def load_obj(file):
    """
    Load to memory an object stored filesystem. This function depends on Dill package

    :param file: file name where the object is stored
    :return: object
    """
    with open(file, 'rb') as _file:
        obj = dill.load(_file)
    return obj


def persist_env(file):
    """
    Persist an entire environment on file. This function depends on Dill package

    :param file: file name to store the environment
    """
    dill.dump_session(file)


def load_env(file):
    dill.load_session(file)



