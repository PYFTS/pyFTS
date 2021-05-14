"""
Common data transformation used on pre and post processing of the FTS
"""

import numpy as np
import pandas as pd
import math
from pyFTS.common.transformations.transformation import Transformation
from pyFTS.common.transformations.differential import Differential
from pyFTS.common.transformations.scale import Scale
from pyFTS.common.transformations.adaptiveexpectation import AdaptiveExpectation 
from pyFTS.common.transformations.boxcox import BoxCox
from pyFTS.common.transformations.roi import ROI
from pyFTS.common.transformations.trend import LinearTrend
from pyFTS.common.transformations.som import SOMTransformation
from pyFTS.common.transformations.autoencoder import AutoencoderTransformation
from pyFTS.common.transformations.normalization import Normalization



