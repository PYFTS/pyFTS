import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from pyFTS.common import Membership, Util


def plot_sets(partitioner, start=0, end=10, step=1, tam=[5, 5], colors=None,
              save=False, file=None, axes=None, data=None, window_size = 1, only_lines=False, legend=True):

    range = np.arange(start,end,step)
    ticks = []
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tam)

    for ct, key in enumerate(partitioner.ordered_sets):
        fset = partitioner.sets[key]
        if not only_lines:
            for t in range:
                tdisp = t - (t % window_size)
                fset.membership(0, tdisp)
                param = fset.perturbated_parameters[str(tdisp)]

                if fset.mf == Membership.trimf:
                    if t == start:
                        line = axes.plot([t, t+1, t], param, label=fset.name)
                        fset.metadata['color'] = line[0].get_color()
                    else:
                        axes.plot([t, t + 1, t], param,c=fset.metadata['color'])

                ticks.extend(["t+"+str(t),""])
        else:
            tmp = []
            for t in range:
                tdisp = t - (t % window_size)
                fset.membership(0, tdisp)
                param = fset.perturbated_parameters[str(tdisp)]
                tmp.append(np.polyval(param, tdisp))
            axes.plot(range, tmp, ls="--", c="blue")

    axes.set_ylabel("Universe of Discourse")
    axes.set_xlabel("Time")
    plt.xticks([k for k in range], ticks, rotation='vertical')
    
    if legend:
        handles0, labels0 = axes.get_legend_handles_labels()
        lgd = axes.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))

    if data is not None:
        axes.plot(np.arange(start, start + len(data), 1), data,c="black")

    if file is not None:
        plt.tight_layout()
        Util.show_and_save_image(fig, file, save)


def plot_sets_conditional(model, data, step=1, size=[5, 5], colors=None,
                          save=False, file=None, axes=None, fig=None):
    range = np.arange(0, len(data), step)
    ticks = []
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=size)

    for t in range:
        model.forecast([data[t]])
        perturb = model.conditional_perturbation_factors(data[t])

        for ct, key in enumerate(model.partitioner.ordered_sets):
            set = model.partitioner.sets[key]
            set.perturbate_parameters(perturb[ct])
            param = set.perturbated_parameters[str(perturb[ct])]

            if set.mf == Membership.trimf:
                if t == 0:
                    line = axes.plot([t, t+1, t], param, label=set.name)
                    set.metadata['color'] = line[0].get_color()
                else:
                    axes.plot([t, t + 1, t], param,c=set.metadata['color'])

            #ticks.extend(["t+"+str(t),""])

    axes.set_ylabel("Universe of Discourse")
    axes.set_xlabel("Time")
    #plt.xticks([k for k in range], ticks, rotation='vertical')

    handles0, labels0 = axes.get_legend_handles_labels()
    lgd = axes.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))

    if data is not None:
        axes.plot(np.arange(0, len(data), 1), data,c="black")

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)
