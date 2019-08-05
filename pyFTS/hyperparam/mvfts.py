"""
Distributed Evolutionary Hyperparameter Optimization (DEHO) for MVFTS

variables: A list of dictionaries, where each dictionary contains
- name: Variable name
- data_label: data label
- type: common | seasonal
- seasonality:

target_variable

genotype: A dictionary containing
- variables: a list with the selected variables, each instance is the index of a variable in variables
- params: a list of dictionaries, where each dictionary contains {mf, npart, partitioner, alpha}

"""


import numpy as np
import pandas as pd
import math
import time
import random
import logging
from pyFTS.common import Util
from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Entropy  # , Huarng
from pyFTS.common import Membership
from pyFTS.models import hofts, ifts, pwfts
from pyFTS.hyperparam import Util as hUtil
from pyFTS.distributed import dispy as dUtil
from pyFTS.hyperparam import Evolutionary
from pyFTS.models.multivariate import mvfts, wmvfts, variable
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime


def genotype(vars, params, tparams, f1=None, f2=None):
    """
    Create the individual genotype

    :param variables: dictionary with explanatory variable names, types, and other parameters
    :param params: dictionary with variable hyperparameters var: {mf, npart, partitioner, alpha}
    :param tparams: dictionary with target variable hyperparameters var: {mf, npart, partitioner, alpha}
    :param f1: accuracy fitness value
    :param f2: parsimony fitness value
    :return: the genotype, a dictionary with all hyperparameters
    """
    ind = dict(
        explanatory_variables=vars,
        explanatory_params=params,
        target_params = tparams,
        f1=f1,
        f2=f2
    )
    return ind


def random_genotype(**kwargs):
    """
    Create random genotype

    :return: the genotype, a dictionary with all hyperparameters
    """
    vars = kwargs.get('variables',None)

    tvar = kwargs.get('target_variable',None)

    l = len(vars)

    nvar = np.random.randint(1,l,1) # the number of variables

    explanatory_variables = np.unique(np.random.randint(0, l, nvar)).tolist() #indexes of the variables

    explanatory_params = []

    for v in explanatory_variables:
        var = vars[v]
        if var['type'] == 'common':
            npart = random.randint(7, 50)
        else:
            npart = var['npart']
        param = {
            'mf': random.randint(1, 4),
            'npart': npart,
            'partitioner': 1, #random.randint(1, 2),
            'alpha': random.uniform(0, .5)
        }
        explanatory_params.append(param)

    target_params = {
            'mf': random.randint(1, 4),
            'npart': random.randint(7, 50),
            'partitioner': 1, #random.randint(1, 2),
            'alpha': random.uniform(0, .5)
        }

    return genotype(
        explanatory_variables,
        explanatory_params,
        target_params
    )


def phenotype(individual, train, fts_method, parameters={}, **kwargs):
    vars = kwargs.get('variables', None)
    tvar = kwargs.get('target_variable', None)

    explanatory_vars = []

    for ct, vix in enumerate(individual['explanatory_variables']):
        var = vars[vix]
        params = individual['explanatory_params'][ct]

        mf = phenotype_mf(params)

        partitioner = phenotype_partitioner(params)

        if var['type'] == 'common':
            tmp = variable.Variable(var['name'], data_label=var['data_label'], alias=var['name'], partitioner=partitioner,
                                   partitioner_specific={'mf': mf}, npart=params['npart'], alpha_cut=params['alpha'],
                                    data=train)
        elif var['type'] == 'seasonal':
            sp = {'seasonality': var['seasonality'], 'mf': mf }
            tmp = variable.Variable(var['name'], data_label=var['data_label'], alias=var['name'],
                                    partitioner=seasonal.TimeGridPartitioner,
                                    partitioner_specific=sp, npart=params['npart'], alpha_cut=params['alpha'],
                                    data=train)

        explanatory_vars.append(tmp)

    tparams = individual['target_params']

    partitioner = phenotype_partitioner(tparams)
    mf = phenotype_mf(tparams)

    target_var = variable.Variable(tvar['name'], data_label=tvar['data_label'], alias=tvar['name'], partitioner=partitioner,
                                   partitioner_specific={'mf': mf}, npart=tparams['npart'], alpha_cut=tparams['alpha'],
                                    data=train)

    explanatory_vars.append(target_var)

    model = fts_method(explanatory_variables=explanatory_vars, target_variable=target_var, **parameters)
    model.fit(train, **parameters)

    return model


def phenotype_partitioner(params):
    if params['partitioner'] == 1:
        partitioner = Grid.GridPartitioner
    elif params['partitioner'] == 2:
        partitioner = Entropy.EntropyPartitioner
    return partitioner


def phenotype_mf(params):
    if params['mf'] == 1:
        mf = Membership.trimf
    elif params['mf'] == 2:
        mf = Membership.trapmf
    elif params['mf'] == 3 and params['partitioner'] != 2:
        mf = Membership.gaussmf
    else:
        mf = Membership.trimf
    return mf


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
    import logging
    from pyFTS.models import hofts, ifts, pwfts
    from pyFTS.common import Util
    from pyFTS.benchmarks import Measures
    from pyFTS.hyperparam.Evolutionary import __measures
    from pyFTS.hyperparam.mvfts import phenotype
    from pyFTS.models.multivariate import mvfts, wmvfts, partitioner, variable, cmvfts,grid, granular, common
    import numpy as np

    window_size = kwargs.get('window_size', 800)
    train_rate = kwargs.get('train_rate', .8)
    increment_rate = kwargs.get('increment_rate', .2)
    fts_method = kwargs.get('fts_method', wmvfts.WeightedMVFTS)
    parameters = kwargs.get('parameters',{})
    tvar = kwargs.get('target_variable', None)

    if individual['f1'] is not None and individual['f2'] is not None:
        return { key: individual[key] for key in __measures }

    errors = []
    lengths = []

    kwargs2 = kwargs.copy()
    kwargs2.pop('fts_method')
    if 'parameters' in kwargs2:
        kwargs2.pop('parameters')

    for count, train, test in Util.sliding_window(dataset, window_size, train=train_rate, inc=increment_rate):

        try:

            model = phenotype(individual, train, fts_method=fts_method, parameters=parameters, **kwargs2)

            forecasts = model.predict(test)

            rmse = Measures.rmse(test[tvar['data_label']].values[model.max_lag:], forecasts[:-1])
            lengths.append(len(model))

            errors.append(rmse)

        except Exception as ex:
            logging.exception("Error")

            lengths.append(np.nan)
            errors.append(np.nan)

    try:
        _rmse = np.nanmean(errors)
        _len = np.nanmean(lengths)

        f1 = np.nansum([.6 * _rmse, .4 * np.nanstd(errors)])
        f2 = np.nansum([.9 * _len, .1 * np.nanstd(lengths)])

        return {'f1': f1, 'f2': f2, 'rmse': _rmse, 'size': _len }
    except Exception as ex:
        logging.exception("Error")
        return {'f1': np.inf, 'f2': np.inf, 'rmse': np.inf, 'size': np.inf}


def crossover(population, **kwargs):
    """
    Crossover operation between two parents

    :param population: the original population
    :return: a genotype
    """
    import random

    n = len(population) - 1

    r1,r2 = 0,0
    while r1 == r2:
        r1 = random.randint(0, n)
        r2 = random.randint(0, n)

    if population[r1]['f1'] < population[r2]['f1']:
        best = population[r1]
        worst = population[r2]
    else:
        best = population[r2]
        worst = population[r1]

    rnd = random.uniform(0, 1)
    nvar = len(best['explanatory_variables']) if rnd < .7 else len(worst['explanatory_variables'])

    explanatory_variables = []
    explanatory_params = []
    for ct in np.arange(nvar):
        if ct < len(best['explanatory_variables']) and ct < len(worst['explanatory_variables']):
            rnd = random.uniform(0, 1)
            ix = best['explanatory_variables'][ct] if rnd < .7 else worst['explanatory_variables'][ct]
        elif ct < len(best['explanatory_variables']):
            ix = best['explanatory_variables'][ct]
        elif ct < len(worst['explanatory_variables']):
            ix = worst['explanatory_variables'][ct]

        if ix in explanatory_variables:
            continue

        if ix in best['explanatory_variables'] and ix in worst['explanatory_variables']:
            bix = best['explanatory_variables'].index(ix)
            wix = worst['explanatory_variables'].index(ix)
            param = crossover_variable_params(best['explanatory_params'][bix], worst['explanatory_params'][wix])
        elif ix in best['explanatory_variables']:
            bix = best['explanatory_variables'].index(ix)
            param = best['explanatory_params'][bix]
        elif ix in worst['explanatory_variables']:
            wix = worst['explanatory_variables'].index(ix)
            param = worst['explanatory_params'][wix]

        explanatory_variables.append(ix)
        explanatory_params.append(param)

    tparams = crossover_variable_params(best['target_params'], worst['target_params'])

    descendent = genotype(explanatory_variables, explanatory_params, tparams, None, None)

    return descendent


def crossover_variable_params(best, worst):
    npart = int(round(.7 * best['npart'] + .3 * worst['npart']))
    alpha = float(.7 * best['alpha'] + .3 * worst['alpha'])
    rnd = random.uniform(0, 1)
    mf = best['mf'] if rnd < .7 else worst['mf']
    rnd = random.uniform(0, 1)
    partitioner = best['partitioner'] if rnd < .7 else worst['partitioner']
    param = {'partitioner': partitioner, 'npart': npart, 'alpha': alpha, 'mf': mf}
    return param

def mutation(individual, **kwargs):
    """
    Mutation operator

    :param individual: an individual genotype
    :param pmut: individual probability o
    :return:
    """

    for ct in np.arange(len(individual['explanatory_variables'])):
        rnd = random.uniform(0, 1)
        if rnd > .5:
            mutate_variable_params(individual['explanatory_params'][ct])

    rnd = random.uniform(0, 1)
    if rnd > .5:
        mutate_variable_params(individual['target_params'])

    individual['f1'] = None
    individual['f2'] = None

    return individual


def mutate_variable_params(param):
    param['npart'] = min(50, max(3, int(param['npart'] + np.random.normal(0, 4))))
    param['alpha'] = min(.5, max(0, param['alpha'] + np.random.normal(0, .5)))
    param['mf'] = random.randint(1, 4)
    param['partitioner'] = random.randint(1, 2)


def execute(datasetname, dataset, **kwargs):
    """
    Batch execution of Distributed Evolutionary Hyperparameter Optimization (DEHO) for monovariate methods

    :param datasetname:
    :param dataset: The time series to optimize the FTS
    :keyword database_file:
    :keyword experiments:
    :keyword distributed:
    :keyword ngen: An integer value with the maximum number of generations, default value: 30
    :keyword mgen: An integer value with the maximum number of generations without improvement to stop, default value 7
    :keyword npop: An integer value with the population size, default value: 20
    :keyword pcross: A float value between 0 and 1 with the probability of crossover, default: .5
    :keyword psel: A float value between 0 and 1 with the probability of selection, default: .5
    :keyword pmut: A float value between 0 and 1 with the probability of mutation, default: .3
    :keyword fts_method: The MVFTS method to optimize
    :keyword parameters: dict with model specific arguments for fts_method
    :keyword elitism: A boolean value indicating if the best individual must always survive to next population
    :keyword selection_operator: a function that receives the whole population and return a selected individual
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

    experiments = kwargs.get('experiments', 30)

    distributed = kwargs.get('distributed', False)

    fts_method = kwargs.get('fts_method', hofts.WeightedHighOrderFTS)
    shortname = str(fts_method.__module__).split('.')[-1]

    kwargs['mutation_operator'] = mutation
    kwargs['crossover_operator'] = crossover
    kwargs['evaluation_operator'] = evaluate
    kwargs['random_individual'] = random_genotype

    if distributed == 'dispy':
        nodes = kwargs.get('nodes', ['127.0.0.1'])
        cluster, http_server = dUtil.start_dispy_cluster(evaluate, nodes=nodes)
        kwargs['cluster'] = cluster

    ret = []
    for i in np.arange(experiments):
        print("Experiment {}".format(i))

        start = time.time()
        ret, statistics = Evolutionary.GeneticAlgorithm(dataset, **kwargs)
        end = time.time()
        ret['time'] = end - start
        experiment = {'individual': ret, 'statistics': statistics}

        ret = process_experiment(shortname, experiment, datasetname)

    if distributed == 'dispy':
        dUtil.stop_dispy_cluster(cluster, http_server)

    return ret


def process_experiment(fts_method, result, datasetname):
    """
    Persist the results of an DEHO execution in sqlite database (best hyperparameters) and json file (generation statistics)

    :param fts_method:
    :param result:
    :param datasetname:
    :param conn:
    :return:
    """

    log_result(datasetname, fts_method, result['individual'])
    persist_statistics(datasetname, result['statistics'])
    return result['individual']


def persist_statistics(datasetname, statistics):
    import json
    with open('statistics_{}.json'.format(datasetname), 'w') as file:
        file.write(json.dumps(statistics))


def log_result(datasetname, fts_method, result):
    import json
    with open('result_{}{}.json'.format(fts_method,datasetname), 'w') as file:
        file.write(json.dumps(result))

        print(result)
