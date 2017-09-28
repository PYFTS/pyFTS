import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS import fts, sfts


class NonStationaryFTS(sfts.SeasonalFTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(NonStationaryFTS, self).__init__(1, "NSFTS " + name, **kwargs)
        self.name = "Non Stationary FTS"
        self.detail = ""
        self.flrgs = {}