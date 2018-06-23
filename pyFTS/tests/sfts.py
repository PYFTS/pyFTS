#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np

import pandas as pd
from pyFTS.common import Util
from pyFTS.benchmarks import benchmarks as bchmk

os.chdir("/home/petronio/Downloads")

data = pd.read_csv("dress_data.csv", sep=",")

#sonda['data'] = pd.to_datetime(sonda['data'])

#data.index = np.arange(0,len(data.index))

data = data["a"].tolist()


from pyFTS.models.seasonal import sfts, cmsfts, SeasonalIndexer

ix = SeasonalIndexer.LinearSeasonalIndexer([7],[1])

from pyFTS.partitioners import Grid

fs = Grid.GridPartitioner(data=data,npart=10)

model = sfts.SeasonalFTS(indexer=ix, partitioner=fs)
#model = cmsfts.ContextualMultiSeasonalFTS(indexer=ix, partitioner=fs)

model.fit(data)

print(model)

print(model.predict(data))

from pyFTS.benchmarks import Measures

Measures.get_point_statistics(data, model)