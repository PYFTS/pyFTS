import numpy as np
import pandas as pd
import math
import time
from functools import reduce
from operator import itemgetter

import random
from pyFTS.common import Util
from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Entropy  # , Huarng
from pyFTS.models import hofts
from pyFTS.common import Membership
from pyFTS.hyperparam import Util as hUtil

from pyFTS.fcm import common, fts


parameters = {}

#
def genotype():
    """
    Create the individual genotype

    :param mf: membership function
    :param npart: number of partitions
    :param partitioner: partitioner method
    :param order: model order
    :param alpha: alpha-cut
    :param lags: array with lag indexes
    :param f1: accuracy fitness value
    :param f2: parsimony fitness value
    :return: the genotype, a dictionary with all hyperparameters
    """
    num_concepts = parameters['num_concepts']
    order = parameters['order']
    ind = dict(
        weights=[np.random.normal(0, 1., (num_concepts,num_concepts)) for k in range(order)],
        bias=[np.random.normal(0, 1., num_concepts) for k in range(order)]
               )
    return ind


def random_genotype():
    """
    Create random genotype

    :return: the genotype, a dictionary with all hyperparameters
    """
    return genotype()


#
def initial_population(n):
    """
    Create a random population of size n

    :param n: the size of the population
    :return: a list with n random individuals
    """
    pop = []
    for i in range(n):
        pop.append(random_genotype())
    return pop


def phenotype(individual, train):
    """
    Instantiate the genotype, creating a fitted model with the genotype hyperparameters

    :param individual: a genotype
    :param train: the training dataset
    :param parameters: dict with model specific arguments for fit method.
    :return: a fitted FTS model
    """
    partitioner = parameters['partitioner']
    order = parameters['order']

    model = fts.FCM_FTS(partitioner=partitioner, order=order)

    model.fcm.weights = individual['weights']
    model.fcm.bias = individual['bias']

    return model


def evaluate(dataset, individual, **kwargs):
    """
    Evaluate an individual using a sliding window cross validation over the dataset.

    :param dataset: Evaluation dataset
    :param individual: genotype to be tested
    :param window_size: The length of scrolling window for train/test on dataset
    :param train_rate: The train/test split ([0,1])
    :param increment_rate: The increment of the scrolling window, relative to the window_size ([0,1])
    :param parameters: dict with model specific arguments for fit method.
    :return: a tuple (len_lags, rmse) with the parsimony fitness value and the accuracy fitness value
    """
    from pyFTS.common import Util
    from pyFTS.benchmarks import Measures
    from pyFTS.fcm.GA import phenotype
    import numpy as np

    window_size = kwargs.get('window_size', 800)
    train_rate = kwargs.get('train_rate', .8)
    increment_rate = kwargs.get('increment_rate', .2)
    #parameters = kwargs.get('parameters',{})


    errors = []

    for count, train, test in Util.sliding_window(dataset, window_size, train=train_rate, inc=increment_rate):

        model = phenotype(individual, train)

        if model is None:
            raise Exception("Phenotype returned None")

        model.uod_clip = False

        forecasts = model.predict(test)

        rmse = Measures.rmse(test[model.max_lag:], forecasts[:-1]) #.get_point_statistics(test, model)

        errors.append(rmse)

    _rmse = np.nanmean(errors)
    _std = np.nanstd(errors)

    #print("EVALUATION {}".format(individual))
    return {'rmse': .6 * _rmse + .4 * _std}


def tournament(population, objective):
    """
    Simple tournament selection strategy.

    :param population: the population
    :param objective: the objective to be considered on tournament
    :return:
    """
    n = len(population) - 1

    r1 = random.randint(0, n) if n > 2 else 0
    r2 = random.randint(0, n) if n > 2 else 1
    ix = r1 if population[r1][objective] < population[r2][objective] else r2
    return population[ix]


def crossover(parents):
    """
    Crossover operation between two parents

    :param parents: a list with two genotypes
    :return: a genotype
    """
    import random

    descendent = genotype()

    for k in range(parameters['order']):
        new_weight = []
        weights1 = parents[0]['weights'][k]
        weights2 = parents[1]['weights'][k]

        for (row, col), a in np.ndenumerate(weights1):
            new_weight.append(.7*weights1[row, col] + .3*weights2[row, col] )

        descendent['weights'][k] = np.array(new_weight).reshape(weights1.shape)

        new_bias = []
        bias1 = parents[0]['bias'][k]
        bias2 = parents[1]['bias'][k]

        for row, a in enumerate(weights1):
            new_bias.append(.7 * bias1[row] + .3 * bias2[row])

        descendent['bias'][k] = np.array(new_bias).reshape(bias1.shape)

    return descendent


def mutation(individual, pmut):
    """
    Mutation operator

    :param population:
    :return:
    """
    import numpy.random

    for k in range(parameters['order']):
        (rows, cols) = individual['weights'][k].shape

        rnd = random.uniform(0, 1)

        if rnd < pmut:

            num_mutations = random.randint(1, parameters['num_concepts']**2)

            for q in np.arange(0, num_mutations):

                row = random.randint(0, rows-1)
                col = random.randint(0, cols-1)

                individual['weights'][k][row, col] += np.random.normal(0, .5, 1)
                individual['weights'][k][row, col] = np.clip(individual['weights'][k][row, col], -1, 1)

                individual['bias'][k][row] += np.random.normal(0, .5, 1)


    return individual


def elitism(population, new_population):
    """
    Elitism operation, always select the best individual of the population and discard the worst

    :param population:
    :param new_population:
    :return:
    """
    population = sorted(population, key=itemgetter('rmse'))
    best = population[0]

    new_population = sorted(new_population, key=itemgetter('rmse'))
    if new_population[0]["rmse"] > best["rmse"]:
        new_population.insert(0,best)

    return new_population


def GeneticAlgorithm(dataset, **kwargs):
    """
    Genetic algoritm for hyperparameter optimization

    :param dataset:
    :param ngen: Max number of generations
    :param mgen: Max number of generations without improvement
    :param npop: Population size
    :param pcruz: Probability of crossover
    :param pmut: Probability of mutation
    :param window_size: The length of scrolling window for train/test on dataset
    :param train_rate: The train/test split ([0,1])
    :param increment_rate: The increment of the scrolling window, relative to the window_size ([0,1])
    :param parameters: dict with model specific arguments for fit method.
    :return: the best genotype
    """

    statistics = []

    ngen = kwargs.get('ngen',30)
    mgen = kwargs.get('mgen', 7)
    npop = kwargs.get('npop',20)
    pcruz = kwargs.get('pcruz',.5)
    pmut = kwargs.get('pmut',.3)
    distributed = kwargs.get('distributed', False)

    if distributed == 'dispy':
        cluster = kwargs.pop('cluster', None)

    collect_statistics = kwargs.get('collect_statistics', True)

    no_improvement_count = 0

    new_population = []

    population = initial_population(npop)

    last_best = population[0]
    best = population[1]

    print("Evaluating initial population {}".format(time.time()))
    if not distributed:
        for individual in population:
            ret = evaluate(dataset, individual, **kwargs)
            individual['rmse'] = ret['rmse']
    elif distributed=='dispy':
        import dispy
        from pyFTS.distributed import dispy as dUtil
        jobs = []
        for ct, individual in enumerate(population):
            job = cluster.submit(dataset, individual, **kwargs)
            job.id = ct
            jobs.append(job)
        for job in jobs:
            result = job()
            if job.status == dispy.DispyJob.Finished and result is not None:
                population[job.id]['rmse'] = result['rmse']
            else:
                print(job.exception)
                print(job.stdout)

    for i in range(ngen):
        print("GENERATION {} {}".format(i, time.time()))

        generation_statistics = {}

        # Selection
        for j in range(int(npop / 2)):
            new_population.append(tournament(population, 'rmse'))
            new_population.append(tournament(population, 'rmse'))

        # Crossover
        new = []
        for j in range(int(npop * pcruz)):
            new.append(crossover(new_population))
        new_population.extend(new)

        # Mutation
        for ct, individual in enumerate(new_population):
            new_population[ct] = mutation(individual, pmut)

        # Evaluation
        if collect_statistics:
            stats = {}
            for key in ['rmse']:
                stats[key] = []

        if not distributed:
            for individual in new_population:
                ret = evaluate(dataset, individual, **kwargs)
                for key in ['rmse']:
                    individual[key] = ret[key]
                    if collect_statistics: stats[key].append(ret[key])

        elif distributed == 'dispy':
            jobs = []

            for ct, individual in enumerate(new_population):
                job = cluster.submit(dataset, individual, **kwargs)
                job.id = ct
                jobs.append(job)
            for job in jobs:
                print('job id {}'.format(job.id))
                result = job()
                if job.status == dispy.DispyJob.Finished and result is not None:
                    for key in ['rmse']:
                        new_population[job.id][key] = result[key]
                        if collect_statistics: stats[key].append(result[key])
                else:
                    print(job.exception)
                    print(job.stdout)


        if collect_statistics:
            mean_stats = {key: np.nanmedian(stats[key]) for key in ['rmse'] }

            generation_statistics['population'] = mean_stats

        # Elitism
        population = elitism(population, new_population)

        population = population[:npop]

        new_population = []

        last_best = best

        best = population[0]

        if collect_statistics:
            generation_statistics['best'] = {key: best[key] for key in ['rmse']}

            statistics.append(generation_statistics)

        if last_best['rmse'] <= best['rmse']:
            no_improvement_count += 1
            print("WITHOUT IMPROVEMENT {}".format(no_improvement_count))
            pmut += .05
        else:
            no_improvement_count = 0
            pcruz = kwargs.get('pcruz', .5)
            pmut = kwargs.get('pmut', .3)
            print(best)

        if no_improvement_count == mgen:
            break


    return best, statistics


def process_experiment(result, datasetname, conn):
    print(result)
    #log_result(conn, datasetname, result['individual'])
    #persist_statistics(result['statistics'])
    return result['individual']


def persist_statistics(statistics):
    import json
    with open('statistics{}.txt'.format(time.time()), 'w') as file:
        file.write(json.dumps(statistics))


def log_result(conn, datasetname, result):
    metrics = ['rmse', 'size', 'time']
    for metric in metrics:
        record = (datasetname, 'Evolutive', 'WHOFTS', None, result['mf'],
                  result['order'], result['partitioner'], result['npart'],
                  result['alpha'], str(result['lags']), metric, result[metric])

        print(record)

        hUtil.insert_hyperparam(record, conn)


def execute(dataset, **kwargs):
    file = kwargs.get('file', 'hyperparam.db')

    conn = hUtil.open_hyperparam_db(file)

    experiments = kwargs.get('experiments', 30)

    distributed = kwargs.get('distributed', False)

    if distributed == 'dispy':
        import dispy
        from pyFTS.distributed import dispy as dUtil
        nodes = kwargs.get('nodes', ['127.0.0.1'])
        cluster, http_server = dUtil.start_dispy_cluster(evaluate, nodes=nodes)
        kwargs['cluster'] = cluster

    ret = []
    for i in np.arange(experiments):
        print("Experiment {}".format(i))

        start = time.time()
        ret, statistics = GeneticAlgorithm(dataset, **kwargs)
        end = time.time()
        ret['time'] = end - start
        experiment = {'individual': ret, 'statistics': statistics}

        ret = process_experiment(experiment, '', conn)

    if distributed == 'dispy':
        dUtil.stop_dispy_cluster(cluster, http_server)

    return ret

