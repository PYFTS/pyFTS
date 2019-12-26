"""
Simple Random Search Hyperparameter Optimization
"""

from pyFTS.hyperparam import Evolutionary
import time

__measures = ['f1', 'f2', 'rmse', 'size']


def execute( dataset, **kwargs):
    """
    Batch execution of Random Search Hyperparameter Optimization

    :param datasetname:
    :param dataset: The time series to optimize the FTS
    :keyword ngen: An integer value with the maximum number of generations, default value: 30
    :keyword mgen: An integer value with the maximum number of generations without improvement to stop, default value 7
    :keyword fts_method: The FTS method to optimize
    :keyword parameters: dict with model specific arguments for fts_method
    :keyword random_individual: create an random genotype
    :keyword evalutation_operator: a function that receives a dataset and an individual and return its fitness
    :keyword mutation_operator: a function that receives one individual and return a changed individual
    :keyword window_size: An integer value with the the length of scrolling window for train/test on dataset
    :keyword train_rate: A float value between 0 and 1 with the train/test split ([0,1])
    :keyword increment_rate: A float value between 0 and 1 with the the increment of the scrolling window,
             relative to the window_size ([0,1])
    :keyword collect_statistics: A boolean value indicating to collect statistics for each generation
    :keyword distributed: A value indicating it the execution will be local and sequential (distributed=False),
             or parallel and distributed (distributed='dispy' or distributed='spark')
    :keyword cluster: If distributed='dispy' the list of cluster nodes, else if distributed='spark' it is the master node
    :return: the best genotype
    """

    ngen = kwargs.get('ngen',30)
    mgen = kwargs.get('mgen', 7)

    kwargs['pmut'] = 1.0

    random_individual = kwargs.get('random_individual', Evolutionary.random_genotype)
    evaluation_operator = kwargs.get('evaluation_operator', Evolutionary.evaluate)
    mutation_operator = kwargs.get('mutation_operator', Evolutionary.mutation)

    no_improvement_count = 0

    individual = random_individual(**kwargs)

    stat = {}

    stat[0] = {}

    ret = evaluation_operator(dataset, individual, **kwargs)
    for key in __measures:
        individual[key] = ret[key]
        stat[0][key] = ret[key]

    print(individual)

    for i in range(1,ngen+1):
        print("GENERATION {} {}".format(i, time.time()))

        new = mutation_operator(individual, **kwargs)
        ret = evaluation_operator(dataset, new, **kwargs)
        new_stat = {}
        for key in __measures:
            new[key] = ret[key]
            new_stat[key] = ret[key]

        print(new)

        if new['f1'] <= individual['f1'] and new['f2'] <= individual['f2']:
            individual = new
            no_improvement_count = 0
            stat[i] = new_stat
            print(individual)
        else:
            stat[i] = stat[i-1]
            no_improvement_count += 1
            print("WITHOUT IMPROVEMENT {}".format(no_improvement_count))

        if no_improvement_count == mgen:
            break

    return individual, stat
