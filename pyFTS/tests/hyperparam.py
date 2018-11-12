
from pyFTS.hyperparam import GridSearch

def get_train_test():
    from pyFTS.data import Malaysia

    ds = Malaysia.get_data('temperature')[:2000]
    # ds =  pd.read_csv('Malaysia.csv',delimiter=',' )[['temperature']].values[:2000].flatten().tolist()
    train = ds[:1000]
    test = ds[1000:]

    return 'Malaysia.temperature', train, test

hyperparams = {
    'order':[1],
    'partitions':[10, 15],
    'partitioner': [1],
    'mf': [1],
    'lags': [1, 2, 3],
    'alpha': [.1, .2, .5]
}

nodes = ['192.168.0.110','192.168.0.106']

ds, train, test = get_train_test()

GridSearch.execute(hyperparams, ds, train, test, nodes=nodes)