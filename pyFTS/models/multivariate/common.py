import numpy as np
import pandas as pd


def fuzzyfy_instance(data_point, var):
    mv = np.array([fs.membership(data_point) for fs in var.partitioner.sets])
    ix = np.ravel(np.argwhere(mv > 0.0))
    sets = [var.partitioner.sets[i] for i in ix]
    return sets


