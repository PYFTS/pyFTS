"""
Activation functions for Time Series Classification
"""

import numpy as np
import math
from pyFTS import *

def scale(dist : dict) -> dict:
  norm = np.sum([v for v in dist.values()])
  return {k : (v / norm) for k,v in dist.items() }

def softmax(dist : dict) -> dict:
  norm = np.sum([np.exp(v) for v in dist.values()])
  return {k : (np.exp(v) / norm) for k,v in dist.items() }

def argmax(dist : dict) -> str:
  mx = np.max([v for v in dist.values()])
  return [k for k,v in dist.items() if v == mx ][0]