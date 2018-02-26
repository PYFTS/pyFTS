import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from pyFTS.common import Membership, Util
from pyFTS.partitioners import Grid,Huarng,FCM,Entropy

all_methods = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner, Huarng.HuarngPartitioner]

mfs = [Membership.trimf, Membership.gaussmf, Membership.trapmf]


def plot_sets(data, sets, titles, tam=[12, 10], save=False, file=None):
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
        ax.set_ylim([0, 1.1])
        ax.set_xlim([minx, maxx])
        for s in sets[k]:
            if s.mf == Membership.trimf:
                ax.plot(s.parameters,[0,1,0])
            elif s.mf == Membership.gaussmf:
                tmpx = [ kk for kk in np.arange(s.lower, s.upper)]
                tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
                ax.plot(tmpx, tmpy)
            elif s.mf == Membership.trapmf:
                ax.plot(s.parameters, [0, 1, 1, 0])

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)


def plot_partitioners(data, objs, tam=[12, 10], save=False, file=None):
    sets = [k.sets for k in objs]
    titles = [k.name for k in objs]
    plot_sets(data,sets,titles,tam,save,file)


def explore_partitioners(data, npart, methods=None, mf=None, tam=[12, 10], save=False, file=None):
    if methods is None:
        methods = all_methods

    if mf is None:
        mf = mfs

    objs = []

    for p in methods:
        for m in mf:
            obj = p(data, npart,m)
            objs.append(obj)

    plot_partitioners(data, objs, tam, save, file)