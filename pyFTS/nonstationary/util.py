import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from pyFTS.common import Membership, Util


def plot_sets(uod, sets, start=0, end=10, tam=[5, 5], colors=None, save=False, file=None):
    ticks = []
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tam)
    for t in np.arange(start,end,1):
        for ct, set in enumerate(sets):
            set.membership(0, t)
            param = set.perturbated_parameters[t]

            if set.mf == Membership.trimf:
                axes.plot([t, t+1, t], param)

        ticks.extend(["t+"+str(t),""])

    axes.set_ylabel("Universe of Discourse")
    axes.set_xlabel("Time")
    plt.xticks([k for k in np.arange(0,2*end,1)], ticks, rotation='vertical')

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)
