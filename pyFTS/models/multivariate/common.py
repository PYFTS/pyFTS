import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet

def fuzzyfy_instance(data_point, var):
    fsets = FuzzySet.fuzzyfy(data_point, var.partitioner, mode='sets', method='fuzzy', alpha_cut=var.alpha_cut)
    return [(var.name, fs) for fs in fsets]



