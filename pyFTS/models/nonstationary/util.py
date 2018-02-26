import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from pyFTS.common import Membership, Util


def plot_sets(sets, start=0, end=10, step=1, tam=[5, 5], colors=None,
              save=False, file=None, axes=None, data=None, window_size = 1, only_lines=False):

    range = np.arange(start,end,step)
    ticks = []
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tam)

    for ct, set in enumerate(sets):
        if not only_lines:
            for t in range:
                tdisp = t - (t % window_size)
                set.membership(0, tdisp)
                param = set.perturbated_parameters[str(tdisp)]

                if set.mf == Membership.trimf:
                    if t == start:
                        line = axes.plot([t, t+1, t], param, label=set.name)
                        set.metadata['color'] = line[0].get_color()
                    else:
                        axes.plot([t, t + 1, t], param,c=set.metadata['color'])

                ticks.extend(["t+"+str(t),""])
        else:
            tmp = []
            for t in range:
                tdisp = t - (t % window_size)
                set.membership(0, tdisp)
                param = set.perturbated_parameters[str(tdisp)]
                tmp.append(np.polyval(param, tdisp))
            axes.plot(range, tmp, ls="--", c="blue")

    axes.set_ylabel("Universe of Discourse")
    axes.set_xlabel("Time")
    plt.xticks([k for k in range], ticks, rotation='vertical')

    handles0, labels0 = axes.get_legend_handles_labels()
    lgd = axes.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))

    if data is not None:
        axes.plot(np.arange(start, start + len(data), 1), data,c="black")

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)


def plot_sets_conditional(model, data, start=0, end=10, step=1, tam=[5, 5], colors=None,
                          save=False, file=None, axes=None):

    range = np.arange(start,end,step)
    ticks = []
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tam)

    for ct, set in enumerate(model.sets):
            for t in range:
                tdisp = model.perturbation_factors(data[t])
                set.perturbate_parameters(tdisp[ct])
                param = set.perturbated_parameters[str(tdisp[ct])]

                if set.mf == Membership.trimf:
                    if t == start:
                        line = axes.plot([t, t+1, t], param, label=set.name)
                        set.metadata['color'] = line[0].get_color()
                    else:
                        axes.plot([t, t + 1, t], param,c=set.metadata['color'])

                ticks.extend(["t+"+str(t),""])

    axes.set_ylabel("Universe of Discourse")
    axes.set_xlabel("Time")
    plt.xticks([k for k in range], ticks, rotation='vertical')

    handles0, labels0 = axes.get_legend_handles_labels()
    lgd = axes.legend(handles0, labels0, loc=2, bbox_to_anchor=(1, 1))

    if data is not None:
        axes.plot(np.arange(start, start + len(data), 1), data,c="black")

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)
