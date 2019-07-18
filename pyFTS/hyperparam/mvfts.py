"""
Distributed Evolutionary Hyperparameter Optimization (DEHO) for MVFTS
"""

import numpy as np
import pandas as pd
import math
import random
from pyFTS.hyperparam import Evolutionary


def genotype(vars, params, f1, f2):
    """
    Create the individual genotype

    :param vars: dictionary with variable names, types, and other parameters
    :param params: dictionary with variable hyperparameters var: {mf, npart, partitioner, alpha}
    :param f1: accuracy fitness value
    :param f2: parsimony fitness value
    :return: the genotype, a dictionary with all hyperparameters
    """
    ind = dict(vars=vars, params=params, f1=f1, f2=f2)
    return ind


def random_genotype(**kwargs):
    """
    Create random genotype

    :return: the genotype, a dictionary with all hyperparameters
    """
    order = random.randint(1, 3)
    lags = [k for k in np.arange(1, order+1)]
    return genotype(
        random.randint(1, 4),
        random.randint(10, 100),
        random.randint(1, 2),
        order,
        random.uniform(0, .5),
        lags,
        None,
        None
    )



def phenotype(individual, train, fts_method, parameters={}):
    pass

