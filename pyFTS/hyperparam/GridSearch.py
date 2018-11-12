
from pyFTS.common import Util, Membership
from pyFTS.models import hofts
from pyFTS.partitioners import Grid, Entropy
from pyFTS.benchmarks import Measures
from pyFTS.hyperparam import Util as hUtil
import numpy as np
import dispy
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


def metodo_cluster(individual, train, test):
    from pyFTS.common import Util, Membership
    from pyFTS.models import hofts
    from pyFTS.partitioners import Grid, Entropy
    from pyFTS.benchmarks import Measures

    if individual['mf'] == 1:
        mf = Membership.trimf
    elif individual['mf'] == 2:
        mf = Membership.trapmf
    elif individual['mf'] == 3 and individual['partitioner'] != 2:
        mf = Membership.gaussmf
    else:
        mf = Membership.trimf

    if individual['partitioner'] == 1:
        partitioner = Grid.GridPartitioner(data=train, npart=individual['npart'], func=mf)
    elif individual['partitioner'] == 2:
        npart = individual['npart'] if individual['npart'] > 10 else 10
        partitioner = Entropy.EntropyPartitioner(data=train, npart=npart, func=mf)


    model = hofts.HighOrderFTS(partitioner=partitioner,
                               lags=individual['lags'],
                               alpha_cut=individual['alpha'],
                               order=individual['order'])

    model.fit(train)

    rmse, mape, u = Measures.get_point_statistics(test, model)

    return individual, rmse


def execute(hyperparams, datasetname, train, test, **kwargs):

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

    hp_values = [
        [v for v in hyperparams[hp]]
        for hp in keys_sorted
    ]

    for instance in product(*hp_values):
        partitions = instance[index['partitions']]
        partitioner = instance[index['partitioner']]
        mf = instance[index['mf']]
        alpha_cut = instance[index['alpha']]
        order = instance[index['order']]
        for lag1 in lags: # o é o lag1
            _lags = [lag1]
            if order > 1:
                for lag2 in lags:  # o é o lag1
                    _lags2 = [lag1, lag1+lag2]
                    if order > 2:
                        for lag3 in lags:  # o é o lag1
                            _lags3 = [lag1, lag1 + lag2, lag1 + lag2+lag3 ]
                            individuals.append(dict_individual(mf, partitioner, partitions, order, _lags3, alpha_cut))
                    else:
                        individuals.append(
                            dict_individual(mf, partitioner, partitions, order, _lags2, alpha_cut))
            else:
                individuals.append(dict_individual(mf, partitioner, partitions, order, _lags, alpha_cut))


    cluster, http_server = Util.start_dispy_cluster(metodo_cluster, nodes=nodes)

    jobs = []

    for ind in individuals:
        job = cluster.submit(ind, train, test)
        jobs.append(job)


    conn = hUtil.open_hyperparam_db('hyperparam.db')

    for job in jobs:
        result, rmse = job()
        if job.status == dispy.DispyJob.Finished and result is not None:
            print(result)

            record = (datasetname, 'GridSearch', 'HOFTS', None, result['mf'],
                      result['order'], result['partitioner'], result['npart'],
                      result['alpha'], str(result['lags']), 'rmse', rmse)

            hUtil.insert_hyperparam(record, conn)

        else:
            print(job.exception)
            print(job.stdout)

    Util.stop_dispy_cluster(cluster, http_server)