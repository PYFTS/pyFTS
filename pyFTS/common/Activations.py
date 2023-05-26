"""
Activation functions for Time Series Classification
"""

import numpy as np
import math
from pyFTS import *

def scale(dist : dict, weights : dict) -> dict:
  norm = np.sum([v for v in dist.values()])
  return {k : ((v * weights[k]) / norm) for k,v in dist.items() }

def softmax(dist : dict, weights : dict) -> dict:
  norm = np.sum([np.exp(v) for v in dist.values()])
  return {k : (np.exp(v * weights[k]) / norm) for k,v in dist.items() }

def argmax(dist : dict, weights : dict) -> str:
  mx = np.max([v * weights[k] for k,v in dist.items()])
  return [k for k,v in dist.items() if v * weights[k] == mx  ][0]
