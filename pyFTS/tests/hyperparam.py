import numpy as np
from pyFTS.hyperparam import GridSearch

def get_train_test():
    from pyFTS.data import Malaysia

    ds = Malaysia.get_data('temperature')[:2000]
    # ds =  pd.read_csv('Malaysia.csv',delimiter=',' )[['temperature']].values[:2000].flatten().tolist()
    train = ds[:1000]
    test = ds[1000:]

    return 'Malaysia.temperature', train, test

hyperparams = {
    'order':[1, 2, 3],
    'partitions': np.arange(10,100,3),
    'partitioner': [1,2],
    'mf': [1, 2, 3, 4],
    'lags': np.arange(1,35,2),
    'alpha': np.arange(0,.5, .05)
}

nodes = ['192.168.0.110','192.168.0.106', '192.168.0.107']

ds, train, test = get_train_test()

GridSearch.execute(hyperparams, ds, train, test, nodes=nodes)
