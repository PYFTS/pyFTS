import numpy as np
from pyFTS.hyperparam import GridSearch

def get_train_test():
    from pyFTS.data import Malaysia

    ds = Malaysia.get_data('temperature')[:1000]
    # ds =  pd.read_csv('Malaysia.csv',delimiter=',' )[['temperature']].values[:2000].flatten().tolist()
    train = ds[:800]
    test = ds[800:]

    return 'Malaysia.temperature', train, test

"""
hyperparams = {
    'order':[1, 2, 3],
    'partitions': np.arange(10,100,3),
    'partitioner': [1,2],
    'mf': [1, 2, 3, 4],
    'lags': np.arange(1,35,2),
    'alpha': np.arange(.0, .5, .05)
}
"""

hyperparams = {
    'order':[3], #[1, 2],
    'partitions': np.arange(10,100,10),
    'partitioner': [1,2],
    'mf': [1] ,#, 2, 3, 4],
    'lags': np.arange(1, 10),
    'alpha': [.0, .3, .5]
}

nodes = ['192.168.0.106', '192.168.0.110', '192.168.0.107']

ds, train, test = get_train_test()

GridSearch.execute(hyperparams, ds, train, test, nodes=nodes)
