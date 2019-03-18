import numpy as np
import pandas as pd
from pyFTS.hyperparam import GridSearch, Evolutionary

def get_dataset():
    from pyFTS.data import SONDA
    #from pyFTS.data import Malaysia

    #data = SONDA.get_data('temperature')[:3000]
    data = pd.read_csv('https://query.data.world/s/6xfb5useuotbbgpsnm5b2l3wzhvw2i', sep=';')
    #data = Malaysia.get_data('temperature')[:1000]

    return 'SONDA.glo_avg', data['glo_avg'].values #train, test
    #return 'Malaysia.temperature', data #train, test


hyperparams = {
    'order':[3],
    'partitions': np.arange(10,100,3),
    'partitioner': [1],
    'mf': [1], #, 2, 3, 4],
    'lags': np.arange(2, 7, 1),
    'alpha': np.arange(.0, .5, .05)
}

'''
hyperparams = {
    'order':[3], #[1, 2],
    'partitions': np.arange(10,100,10),
    'partitioner': [1,2],
    'mf': [1] ,#, 2, 3, 4],
    'lags': np.arange(1, 10),
    'alpha': [.0, .3, .5]
}
'''
nodes = ['192.168.0.106', '192.168.0.110', '192.168.0.107']

datsetname, dataset  = get_dataset()

#GridSearch.execute(hyperparams, datsetname, dataset, nodes=nodes,
#                   window_size=10000, train_rate=.9, increment_rate=1,)

ret = Evolutionary.execute(datsetname, dataset,
                            ngen=30, npop=20, pcruz=.5, pmut=.3,
                            window_size=10000, train_rate=.9, increment_rate=1,
                            experiments=1,
                            distributed='dispy', nodes=nodes)

#res = GridSearch.cluster_method({'mf':1, 'partitioner': 1, 'npart': 10, 'lags':[1], 'alpha': 0.0, 'order': 1},
#                          dataset, window_size = 10000, train_rate = .9, increment_rate = 1)

#print(res)

#Evolutionary.cluster_method(dataset, 70, 20, .8, .3, 1)

"""
from pyFTS.models import hofts
from pyFTS.partitioners import Grid
from pyFTS.benchmarks import Measures

fs = Grid.GridPartitioner(data=dataset[:800], npart=30)

model = hofts.WeightedHighOrderFTS(partitioner=fs, order=2)

model.fit(dataset[:800])

model.predict(dataset[800:1000])

Measures.get_point_statistics(dataset[800:1000], model)

print(model)


ret = Evolutionary.execute(datsetname, dataset,
                     ngen=30, npop=20, pcruz=.5, pmut=.3,
                     window_size=800, experiments=30)
                     parameters={'distributed': 'spark', 'url': 'spark://192.168.0.106:7077'})

print(ret)


from pyFTS.hyperparam import Evolutionary


from pyFTS.data import SONDA

data = np.array(SONDA.get_data('glo_avg'))

data =  data[~(np.isnan(data) | np.equal(data, 0.0))]

dataset = data[:1000000]

del(data)



import pandas as pd
df = pd.read_csv('https://query.data.world/s/i7eb73c4rluf2luasppsyxaurx5ol7', sep=';')
dataset = df['glo_avg'].values

from pyFTS.models import hofts
from pyFTS.partitioners import Grid
from pyFTS.benchmarks import Measures

from time import  time

t1 = time()





Evolutionary.execute('SONDA', dataset,
                     ngen=20, mgen=5, npop=15, pcruz=.5, pmut=.3,
                     window_size=35000, train_rate=.6, increment_rate=1,
                     collect_statistics=True, experiments=5)
                     #distributed='dispy', nodes=['192.168.0.110','192.168.0.106','192.168.0.107'])

t2 = time()

print(t2 - t1)
"""
