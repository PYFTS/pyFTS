#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FLR, FuzzySet, SortedCollection
from pyFTS import fts, chen, cheng, hofts, hwang, ismailefendi, sadaei, song, yu, sfts
from pyFTS.benchmarks import arima, quantreg
from pyFTS.common import Transformations, Util as cUtil
import scipy.stats as st
from pyFTS.ensemble import ensemble
from pyFTS.models import msfts
from pyFTS.probabilistic import ProbabilityDistribution, kde
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing


def train_individual_model(partitioner, train_data, indexer):
    pttr = str(partitioner.__module__).split('.')[-1]
    _key = "msfts_" + pttr + str(partitioner.partitions) + "_" + indexer.name

    model = msfts.MultiSeasonalFTS(_key, indexer=indexer)
    model.appendTransformation(partitioner.transformation)
    model.train(train_data, partitioner.sets, order=1)

    cUtil.persist_obj(model, "models/"+_key+".pkl")

    print(_key)

    return model


class SeasonalEnsembleFTS(ensemble.EnsembleFTS):
    def __init__(self, name, **kwargs):
        super(SeasonalEnsembleFTS, self).__init__(name="Seasonal Ensemble FTS", **kwargs)
        self.min_order = 1
        self.indexers = []
        self.partitioners = []
        self.is_multivariate = True
        self.has_seasonality = True
        self.has_probability_forecasting = True

    def train(self, data, sets, order=1, parameters=None):
        self.original_max = max(data)
        self.original_min = min(data)

        num_cores = multiprocessing.cpu_count()

        pool = {}
        count = 0
        for ix in self.indexers:
            for pt in self.partitioners:
                pool[count] = {'ix': ix, 'pt': pt}

        results = Parallel(n_jobs=num_cores)(delayed(train_individual_model)(deepcopy(pool[m]['pt']), deepcopy(data), deepcopy(pool[m]['ix'])) for m in pool.keys())

        for tmp in results:
            self.appendModel(tmp)

    def forecastDistribution(self, data, **kwargs):

        ret = []

        h = kwargs.get("h",10)

        for k in data:

            tmp = self.get_models_forecasts(k)

            dist = ProbabilityDistribution.ProbabilityDistribution("KDE",h)

            ret.append(dist)

        return ret