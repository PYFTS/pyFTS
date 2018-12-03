import numpy as np
import pandas as pd
import math
import time
from functools import reduce
from operator import itemgetter
import dispy

import random
from pyFTS.common import Util
from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Entropy  # , Huarng
from pyFTS.models import hofts
from pyFTS.common import Membership
from pyFTS.hyperparam import Util as hUtil


# Gera indivíduos após operadores
def genotype(mf, npart, partitioner, order, alpha, lags, len_lags, rmse):
    ind = dict(mf=mf, npart=npart, partitioner=partitioner, order=order, alpha=alpha, lags=lags, len_lags=len_lags,
               rmse=rmse)
    return ind


# Gera indivíduos
def random_genotype():
    order = random.randint(1, 3)
    return genotype(
        random.randint(1, 4),
        random.randint(10, 100),
        random.randint(1, 2),
        order,
        random.uniform(0, .5),
        sorted(random.sample(range(1, 50), order)),
        [],
        []
    )


# Gera uma população de tamanho n
def initial_population(n):
    pop = []
    for i in range(n):
        pop.append(random_genotype())
    return pop


# Função de avaliação
def phenotype(individual, train):
    try:
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
            partitioner = Entropy.EntropyPartitioner(data=train, npart=individual['npart'], func=mf)

        model = hofts.WeightedHighOrderFTS(partitioner=partitioner,
                                           lags=individual['lags'],
                                           alpha_cut=individual['alpha'],
                                           order=individual['order'])

        model.fit(train)

        return model

    except Exception as ex:
        print("EXCEPTION!", str(ex), str(individual))
        return None


def evaluation1(dataset, individual):
    from pyFTS.common import Util
    from pyFTS.benchmarks import Measures

    try:
        results = []
        lengths = []

        for count, train, test in Util.sliding_window(dataset, 800, train=.8, inc=.25):
            model = phenotype(individual, train)

            if model is None:
                return (None)

            rmse, _, _ = Measures.get_point_statistics(test, model)
            lengths.append(len(model))

            results.append(rmse)

            _lags = sum(model.lags) * 100

            rmse = np.nansum([.6 * np.nanmean(results), .4 * np.nanstd(results)])
            len_lags = np.nansum([.4 * np.nanmean(lengths), .6 * _lags])

        return len_lags, rmse

    except Exception as ex:
        print("EXCEPTION!", str(ex), str(individual))
        return np.inf


def tournament(population, objective):
    n = len(population) - 1

    r1 = random.randint(0, n) if n > 2 else 0
    r2 = random.randint(0, n) if n > 2 else 1
    ix = r1 if population[r1][objective] < population[r2][objective] else r2
    return population[ix]


def selection1(population):
    pais = []
    prob = .8

    # for i in range(len(population)):
    pai1 = tournament(population, 'rmse')
    pai2 = tournament(population, 'rmse')

    finalista = tournament([pai1, pai2], 'len_lags')

    return finalista


def lag_crossover2(best, worst):
    order = int(round(.7 * best['order'] + .3 * worst['order']))
    lags = []

    min_order = min(best['order'], worst['order'])

    max_order = best if best['order'] > min_order else worst

    for k in np.arange(0, order):
        if k < min_order:
            lags.append(int(round(.7 * best['lags'][k] + .3 * worst['lags'][k])))
        else:
            lags.append(max_order['lags'][k])

    for k in range(1, order):
        while lags[k - 1] >= lags[k]:
            lags[k] += random.randint(1, 10)

    return order, lags


# Cruzamento
def crossover(pais):
    import random

    if pais[0]['rmse'] < pais[1]['rmse']:
        best = pais[0]
        worst = pais[1]
    else:
        best = pais[1]
        worst = pais[0]

    npart = int(round(.7 * best['npart'] + .3 * worst['npart']))
    alpha = float(.7 * best['alpha'] + .3 * worst['alpha'])

    rnd = random.uniform(0, 1)
    mf = best['mf'] if rnd < .7 else worst['mf']

    rnd = random.uniform(0, 1)
    partitioner = best['partitioner'] if rnd < .7 else worst['partitioner']

    order, lags = lag_crossover2(best, worst)

    rmse = []
    len_lags = []

    filho = genotype(mf, npart, partitioner, order, alpha, lags, len_lags, rmse)

    return filho


# Mutação | p é a probabilidade de mutação

def mutation_lags(lags, order):
    new = sorted(random.sample(range(1, 50), order))
    for lag in np.arange(len(lags) - 1):
        new[lag] = min(50, max(1, int(lags[lag] + np.random.normal(0, 0.5))))

    if order > 1:
        for k in np.arange(1, order):
            while new[k] <= new[k - 1]:
                new[k] = int(new[k] + np.random.randint(1, 5))

    return new


def mutation(individual):
    import numpy.random
    individual['npart'] = min(50, max(3, int(individual['npart'] + np.random.normal(0, 2))))
    individual['alpha'] = min(.5, max(0, individual['alpha'] + np.random.normal(0, .1)))
    individual['mf'] = random.randint(1, 2)
    individual['partitioner'] = random.randint(1, 2)
    individual['order'] = min(5, max(1, int(individual['order'] + np.random.normal(0, 0.5))))
    # Chama a função mutation_lags
    individual['lags'] = mutation_lags( individual['lags'],  individual['order'])
    #individual['lags'] = sorted(random.sample(range(1, 50), individual['order']))

    return individual


# Elitismo
def elitism(population, new_population):
    # Pega melhor indivíduo da população corrente
    population = sorted(population, key=itemgetter('rmse'))
    best = population[0]

    # Ordena a nova população e insere o melhor1 no lugar do pior
    new_population = sorted(new_population, key=itemgetter('rmse'))
    new_population[-1] = best

    # Ordena novamente e pega o melhor
    new_population = sorted(new_population, key=itemgetter('rmse'))

    return new_population


def genetico(dataset, ngen, npop, pcruz, pmut, option=1):
    new_populacao = populacao_nova = []
    # Gerar população inicial
    populacao = initial_population(npop)

    # Avaliar população inicial
    result = [evaluation1(dataset, k) for k in populacao]

    for i in range(npop):
        if option == 1:
            populacao[i]['len_lags'], populacao[i]['rmse'] = result[i]
        else:
            populacao[i]['rmse'] = result[i]

    # Gerações
    for i in range(ngen):
        # Iteração para gerar a nova população
        for j in range(int(npop / 2)):
            # Selecao de pais
            pais = []
            pais.append(selection1(populacao))
            pais.append(selection1(populacao))

            # Cruzamento com probabilidade pcruz
            rnd = random.uniform(0, 1)
            filho1 = crossover(pais) if pcruz > rnd else pais[0]
            rnd = random.uniform(0, 1)
            filho2 = crossover(pais) if pcruz > rnd else pais[1]

            # Mutação com probabilidade pmut
            rnd = random.uniform(0, 1)
            filho11 = mutation(filho1) if pmut > rnd else filho1
            rnd = random.uniform(0, 1)
            filho22 = mutation(filho2) if pmut > rnd else filho2

            # Insere filhos na nova população
            new_populacao.append(filho11)
            new_populacao.append(filho22)

        result = [evaluation1(dataset, k) for k in new_populacao]

        for i in range(len(new_populacao)):
            new_populacao[i]['len_lags'], new_populacao[i]['rmse'] = result[i]

        populacao = elitism(populacao, new_populacao)

        new_populacao = []

    melhorT = sorted(populacao, key=lambda item: item['rmse'])[0]

    return melhorT


def cluster_method(dataset, ngen, npop, pcruz, pmut, option=1):
    print(ngen, npop, pcruz, pmut, option)

    from pyFTS.hyperparam.Evolutionary import genetico

    inicio = time.time()
    ret = genetico(dataset, ngen, npop, pcruz, pmut, option)
    fim = time.time()
    ret['time'] = fim - inicio
    ret['size'] = ret['len_lags']
    return ret


def process_jobs(jobs, datasetname, conn):
    for job in jobs:
        result = job()
        if job.status == dispy.DispyJob.Finished and result is not None:
            print("Processing result of {}".format(result))

            metrics = ['rmse', 'size', 'time']

            for metric in metrics:
                record = (datasetname, 'Evolutive', 'WHOFTS', None, result['mf'],
                          result['order'], result['partitioner'], result['npart'],
                          result['alpha'], str(result['lags']), metric, result[metric])

                print(record)

                hUtil.insert_hyperparam(record, conn)
                

        else:
            print(job.exception)
            print(job.stdout)


def execute(datasetname, dataset, **kwargs):
    nodes = kwargs.get('nodes', ['127.0.0.1'])

    cluster, http_server = Util.start_dispy_cluster(cluster_method, nodes=nodes)
    conn = hUtil.open_hyperparam_db('hyperparam.db')

    ngen = kwargs.get('ngen', 70)
    npop = kwargs.get('npop', 20)
    pcruz = kwargs.get('pcruz', .8)
    pmut = kwargs.get('pmut', .2)
    option = kwargs.get('option', 1)

    jobs = []

    for i in range(kwargs.get('experiments', 30)):
        print("Experiment {}".format(i))
        job = cluster.submit(dataset, ngen, npop, pcruz, pmut, option)
        jobs.append(job)

    process_jobs(jobs, datasetname, conn)

    Util.stop_dispy_cluster(cluster, http_server)
