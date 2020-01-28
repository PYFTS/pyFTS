
from pyFTS.common import Util, Membership
from pyFTS.models import hofts
from pyFTS.partitioners import Grid, Entropy
from pyFTS.benchmarks import Measures
from pyFTS.hyperparam import Util as hUtil

import numpy as np
from itertools import product


def dict_individual(mf, partitioner, partitions, order, lags, alpha_cut):
    return {
        'mf': mf,
        'partitioner': partitioner,
        'npart': partitions,
        'alpha': alpha_cut,
        'order': order,
        'lags': lags
    }


def cluster_method(individual, dataset, **kwargs):
    from pyFTS.common import Util, Membership
    from pyFTS.models import hofts
    from pyFTS.partitioners import Grid, Entropy
    from pyFTS.benchmarks import Measures
    import numpy as np

    if individual['mf'] == 1:
        mf = Membership.trimf
    elif individual['mf'] == 2:
        mf = Membership.trapmf
    elif individual['mf'] == 3 and individual['partitioner'] != 2:
        mf = Membership.gaussmf
    else:
        mf = Membership.trimf

    window_size = kwargs.get('window_size', 800)
    train_rate = kwargs.get('train_rate', .8)
    increment_rate = kwargs.get('increment_rate', .2)
    parameters = kwargs.get('parameters', {})

    errors = []
    sizes = []

    for count, train, test in Util.sliding_window(dataset, window_size, train=train_rate, inc=increment_rate):

        if individual['partitioner'] == 1:
            partitioner = Grid.GridPartitioner(data=train, npart=individual['npart'], func=mf)
        elif individual['partitioner'] == 2:
            npart = individual['npart'] if individual['npart'] > 10 else 10
            partitioner = Entropy.EntropyPartitioner(data=train, npart=npart, func=mf)

        model = hofts.WeightedHighOrderFTS(partitioner=partitioner,
                                   lags=individual['lags'],
                                   alpha_cut=individual['alpha'],
                                   order=individual['order'])
        model.fit(train)

        forecasts = model.predict(test)

        #rmse, mape, u = Measures.get_point_statistics(test, model)
        rmse = Measures.rmse(test[model.max_lag:], forecasts)

        size = len(model)

        errors.append(rmse)
        sizes.append(size)

    return {'parameters': individual, 'rmse': np.nanmean(errors), 'size': np.nanmean(size)}


def process_jobs(jobs, datasetname, conn):
    from pyFTS.distributed import dispy as dUtil
    import dispy
    for ct, job in enumerate(jobs):
        print("Processing job {}".format(ct))
        result = job()
        if job.status == dispy.DispyJob.Finished and result is not None:
            print("Processing result of {}".format(result))
            
            metrics = {'rmse': result['rmse'], 'size': result['size']}
            
            for metric in metrics.keys():

                param = result['parameters']

                record = (datasetname, 'GridSearch', 'WHOFTS', None, param['mf'],
                          param['order'], param['partitioner'], param['npart'],
                          param['alpha'], str(param['lags']), metric, metrics[metric])


                hUtil.insert_hyperparam(record, conn)

        else:
            print(job.exception)
            print(job.stdout)
    

def execute(hyperparams, datasetname, dataset, **kwargs):
    from pyFTS.distributed import dispy as dUtil
    import dispy

    nodes = kwargs.get('nodes',['127.0.0.1'])

    individuals = []

    if 'lags' in hyperparams:
        lags = hyperparams.pop('lags')
    else:
        lags = [k for k in np.arange(50)]

    keys_sorted = [k for k in sorted(hyperparams.keys())]

    index = {}
    for k in np.arange(len(keys_sorted)):
        index[keys_sorted[k]] = k
        
    print("Evaluation order: \n {}".format(index))

    hp_values = [
        [v for v in hyperparams[hp]]
        for hp in keys_sorted
    ]
    
    print("Evaluation values: \n {}".format(hp_values))
    
    cluster, http_server = dUtil.start_dispy_cluster(cluster_method, nodes=nodes)
    file = kwargs.get('file', 'hyperparam.db')

    conn = hUtil.open_hyperparam_db(file)

    for instance in product(*hp_values):
        partitions = instance[index['partitions']]
        partitioner = instance[index['partitioner']]
        mf = instance[index['mf']]
        alpha_cut = instance[index['alpha']]
        order = instance[index['order']]
        count = 0
        for lag1 in lags: # o é o lag1
            _lags = [lag1]
            count += 1
            if order > 1:
                for lag2 in lags:  # o é o lag1
                    _lags2 = [lag1, lag1+lag2]
                    count += 1
                    if order > 2:
                        for lag3 in lags:  # o é o lag1
                            count += 1
                            _lags3 = [lag1, lag1 + lag2, lag1 + lag2+lag3 ]
                            individuals.append(dict_individual(mf, partitioner, partitions, order, _lags3, alpha_cut))
                    else:
                        individuals.append(
                            dict_individual(mf, partitioner, partitions, order, _lags2, alpha_cut))
            else:
                individuals.append(dict_individual(mf, partitioner, partitions, order, _lags, alpha_cut))
                
            if count > 10:
                jobs = []

                for ind in individuals:
                    print("Testing individual {}".format(ind))
                    job = cluster.submit(ind, dataset, **kwargs)
                    jobs.append(job)
                    
                process_jobs(jobs, datasetname, conn)
                
                count = 0
                
                individuals = []

    dUtil.stop_dispy_cluster(cluster, http_server)
