import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyFTS.common import Membership, Util


def plotSets(data, sets, titles, tam=[12, 10], save=False, file=None):
    num = len(sets)
    #fig = plt.figure(figsize=tam)
    maxx = max(data)
    minx = min(data)
    #h = 1/num
    #print(h)
    fig, axes = plt.subplots(nrows=num, ncols=1,figsize=tam)
    for k in np.arange(0,num):
        #ax = fig.add_axes([0.05, 1-(k*h), 0.9, h*0.7])  # left, bottom, width, height
        ax = axes[k]
        ax.set_title(titles[k])
        ax.set_ylim([0, 1])
        ax.set_xlim([minx, maxx])
        for s in sets[k]:
            if s.mf == Membership.trimf:
                ax.plot([s.parameters[0],s.parameters[1],s.parameters[2]],[0,1,0])
            elif s.mf == Membership.gaussmf:
                tmpx = [ kk for kk in np.arange(s.lower, s.upper)]
                tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
                ax.plot(tmpx, tmpy)

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)