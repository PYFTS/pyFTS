import numpy as np
import pandas as pd
from enum import Enum
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner, Grid


class DateTime(Enum):
    year = 1
    month = 2
    day_of_month = 3
    day_of_year = 4
    day_of_week = 5
    hour = 6
    minute = 7
    second = 8


def strip_datepart(self, date, date_part):
    if date_part == DateTime.year:
        tmp = date.year
    elif date_part == DateTime.month:
        tmp = date.month
    elif date_part == DateTime.day_of_year:
        tmp = date.timetuple().tm_yday
    elif date_part == DateTime.day_of_month:
        tmp = date.day
    elif date_part == DateTime.day_of_week:
        tmp = date.weekday()
    elif date_part == DateTime.hour:
        tmp = date.hour
    elif date_part == DateTime.minute:
        tmp = date.minute
    elif date_part == DateTime.second:
        tmp = date.second

    return tmp


class FuzzySet(FuzzySet.FuzzySet):
    """
    Temporal/Seasonal Fuzzy Set
    """

    def __init__(self, datepart, name, mf, parameters, centroid):
        super(FuzzySet, self).__init__(name, mf, parameters, centroid)
        self.datepart = datepart

    def membership(self, x):
        dp = strip_datepart(x, self.datepart)
        return self.mf.membership(dp)
