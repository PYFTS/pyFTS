from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
import  numpy as np

from pyFTS.common import Membership, Util
from pyFTS.partitioners import Grid,Huarng,FCM,Entropy
from pyFTS.partitioners import Util


def explore_partitioners(data, npart, methods=None, mf=None, tam=[12, 10], save=False, file=None):
    all_methods = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]
    mfs = [Membership.trimf, Membership.gaussmf, Membership.trapmf]

    if methods is None:
        methods = all_methods

    if mf is None:
        mf = mfs

    num_cores = multiprocessing.cpu_count()

    objs = []
    for method in methods:
        print(str(method))
        tmp = Parallel(n_jobs=num_cores)(delayed(method)(deepcopy(data), npart, m) for m in mf)
        objs.append(tmp)

    objs = np.ravel(objs).tolist()

    Util.plot_partitioners(data, objs, tam, save, file)

