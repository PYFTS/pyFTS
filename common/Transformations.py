import numpy as np
from pyFTS import *

def differential(original):
    n = len(original)
    diff = [ original[t-1]-original[t] for t in np.arange(1,n) ]
    diff.insert(0,0)
    return np.array(diff)