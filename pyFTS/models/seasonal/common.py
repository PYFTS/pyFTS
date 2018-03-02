import numpy as np
import pandas as pd
from enum import Enum
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner, Grid


class DateTime(Enum):
    year = 1
    month = 12
    day_of_month = 30
    day_of_year = 364
    day_of_week = 7
    hour = 6
    minute = 7
    second = 8
    hour_of_day = 24
    hour_of_week = 168
    hour_of_month = 744
    hour_of_year = 8736
    minute_of_hour = 60
    minute_of_day = 1440
    minute_of_week = 10080
    minute_of_month = 44640
    minute_of_year = 524160
    second_of_minute = 60.00001
    second_of_hour = 3600
    second_of_day = 86400


def strip_datepart(date, date_part):
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
    elif date_part == DateTime.hour or date_part == DateTime.hour_of_day:
        tmp = date.hour
    elif date_part == DateTime.hour_of_week:
        wk = (date.weekday()-1) * 24
        tmp = date.hour + wk
    elif date_part == DateTime.hour_of_month:
        wk = (date.day-1) * 24
        tmp = date.hour + wk
    elif date_part == DateTime.hour_of_year:
        wk = (date.timetuple().tm_yday-1) * 24
        tmp = date.hour + wk
    elif date_part == DateTime.minute or date_part == DateTime.minute_of_hour:
        tmp = date.minute
    elif date_part == DateTime.minute_of_day:
        wk = date.hour * 60
        tmp = date.minute + wk
    elif date_part == DateTime.minute_of_week:
        wk1 = (date.weekday()-1) * 1440 #24 * 60
        wk2 = date.hour * 60
        tmp = date.minute + wk1 + wk2
    elif date_part == DateTime.minute_of_month:
        wk1 = (date.day - 1) * 1440 #24 * 60
        wk2 = date.hour * 60
        tmp = date.minute + wk1 + wk2
    elif date_part == DateTime.minute_of_year:
        wk1 = (date.timetuple().tm_yday - 1) * 1440 #24 * 60
        wk2 = date.hour * 60
        tmp = date.minute + wk1 + wk2
    elif date_part == DateTime.second or date_part == DateTime.second_of_minute:
        tmp = date.second
    elif date_part == DateTime.second_of_hour:
        wk1 = date.minute * 60
        tmp = date.second + wk1
    elif date_part == DateTime.second_of_day:
        wk1 = date.hour * 3600 #60 * 60
        wk2 = date.minute * 60
        tmp = date.second + wk1 + wk2
    else:
        raise Exception("Unknown DateTime value!")

    return tmp


class FuzzySet(FuzzySet.FuzzySet):
    """
    Temporal/Seasonal Fuzzy Set
    """

    def __init__(self, datepart, name, mf, parameters, centroid, alpha=1.0, **kwargs):
        super(FuzzySet, self).__init__(name, mf, parameters, centroid, alpha, type = 'datetime', **kwargs)
        self.datepart = datepart
        self.type = 'seasonal'

    def membership(self, x):
        dp = strip_datepart(x, self.datepart)
        return self.mf(dp, self.parameters) * self.alpha