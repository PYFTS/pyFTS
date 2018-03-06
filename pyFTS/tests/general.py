#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.common import Transformations

from pyFTS.data import INMET

print(INMET.get_dataframe())
