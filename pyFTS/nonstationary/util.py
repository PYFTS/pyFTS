import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from pyFTS.common import Membership, Util


def plot_sets(uod, sets, num=10, tam=[5, 5], colors=None, save=False, file=None):
    ticks = []
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tam)
    for t in np.arange(0,num,1):
        for ct, set in enumerate(sets):
            x = [2*t + set.membership(v, t) for v in uod]
            if colors is not None: c = colors[ct]
            axes.plot(x, uod, c=c)
        ticks.extend(["t+"+str(t),""])

    axes.set_ylabel("Universe of Discourse")
    axes.set_xlabel("Time")
    plt.xticks([k for k in np.arange(0,2*num,1)], ticks, rotation='vertical')

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)
