import numpy as np
import pandas as pd

from pyFTS.data import Enrollments, TAIEX
from pyFTS.partitioners import Grid, Simple
from pyFTS.models.multivariate import partitioner as mv_partitioner
from pyFTS.models import hofts

from pyspark import SparkConf
from pyspark import SparkContext

import os
# make sure pyspark tells workers to use python3 not 2 if both are installed
SPARK_ADDR = 'spark://192.168.0.110:7077'

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'



def get_partitioner(shared_partitioner, type='common', variables=[]):
    """

    :param part:
    :return:
    """
    if type=='common':
        fs_tmp = Simple.SimplePartitioner()

    for fset in shared_partitioner.value.keys():
        fz = shared_partitioner.value[fset]
        if type=='common':
            fs_tmp.append_complex(fz)
        elif type == 'multivariate':
            fs_tmp.append(fz)

    return fs_tmp


def get_clustered_partitioner(explanatory_variables, target_variable, **parameters):
    from pyFTS.models.multivariate.common import MultivariateFuzzySet
    fs_tmp = mv_partitioner.MultivariatePartitioner(explanatory_variables=explanatory_variables,
                                           target_variable=target_variable)
    for tmp in parameters['partitioner_names'].value:
        fs = MultivariateFuzzySet(target_variable=target_variable)
        for var, fset in parameters['partitioner_{}'.format(tmp)].value:
            fs.append_set(var, fset)
        fs_tmp.append(fs)

    fs_tmp.build_index()

    return fs_tmp


def get_variables(**parameters):
    explanatory_variables = []
    target_variable = None
    for name in parameters['variables'].value:
        from pyFTS.models.multivariate import common, variable
        var = variable.Variable(name,
                                type=parameters['{}_type'.format(name)].value,
                                data_label=parameters['{}_label'.format(name)].value,
                                alpha_cut=parameters['{}_alpha'.format(name)].value,
                                #data_type=parameters['{}_data_type'.format(name)].value,
                                #mask=parameters['{}_mask'.format(name)].value,
                                )
        var.partitioner = get_partitioner(parameters['{}_partitioner'.format(name)])
        var.partitioner.type = parameters['{}_partitioner_type'.format(name)].value

        explanatory_variables.append(var)

        if var.name == parameters['target'].value:
            target_variable = var

    return (explanatory_variables, target_variable)


def slave_train_univariate(data, **parameters):
    """

    :param data:
    :return:
    """

    if parameters['type'].value == 'common':

        if parameters['order'].value > 1:
            model = parameters['method'].value(partitioner=get_partitioner(parameters['partitioner']),
                                               order=parameters['order'].value, alpha_cut=parameters['alpha_cut'].value,
                                               lags=parameters['lags'].value)
        else:
            model = parameters['method'].value(partitioner=get_partitioner(parameters['partitioner']),
                                               alpha_cut=parameters['alpha_cut'].value)

        ndata = [k for k in data]

    else:
        pass

    model.train(ndata)

    return [(k, model.flrgs[k]) for k in model.flrgs.keys()]


def slave_train_multivariate(data, **parameters):
    explanatory_variables, target_variable = get_variables(**parameters)
    #vars = [(v.name, v.name) for v in explanatory_variables]

    #return [('vars', vars), ('target',[target_variable.name])]

    if parameters['type'].value == 'clustered':
        fs = get_clustered_partitioner(explanatory_variables, target_variable, **parameters)
        model = parameters['method'].value(explanatory_variables=explanatory_variables,
                                           target_variable=target_variable,
                                           partitioner=fs,
                                           order=parameters['order'].value,
                                           alpha_cut=parameters['alpha_cut'].value,
                                           lags=parameters['lags'].value)
    else:

        if parameters['order'].value > 1:
            model = parameters['method'].value(explanatory_variables=explanatory_variables,
                                               target_variable=target_variable,
                                               order=parameters['order'].value,
                                               alpha_cut=parameters['alpha_cut'].value,
                                               lags=parameters['lags'].value)
        else:
            model = parameters['method'].value(explanatory_variables=explanatory_variables,
                                               target_variable=target_variable,
                                               alpha_cut=parameters['alpha_cut'].value)



    rows = [k for k in data]
    ndata = pd.DataFrame.from_records(rows, columns=parameters['columns'].value)

    model.train(ndata)

    if parameters['type'].value == 'clustered':
        counts = [(fset, count) for fset,count in model.partitioner.count.items()]
        flrgs = [(k, v) for k,v in model.flrgs.items()]

        return [('counts', counts), ('flrgs', flrgs)]
    else:
        return [(k, v) for k,v in model.flrgs.items()]


def distributed_train(model, data, url=SPARK_ADDR, app='pyFTS'):
    """


    :param model:
    :param data:
    :param url:
    :param app:
    :return:
    """

    conf = SparkConf()
    conf.setMaster(url)
    conf.setAppName(app)
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    conf.set("spark.memory.offHeap.enabled",True)
    conf.set("spark.memory.offHeap.size","16g")
    parameters = {}

    with SparkContext(conf=conf) as context:

        nodes = context.defaultParallelism

        if not model.is_multivariate:
            parameters['type'] = context.broadcast('common')
            parameters['partitioner'] = context.broadcast(model.partitioner.sets)
            parameters['alpha_cut'] = context.broadcast(model.alpha_cut)
            parameters['order'] = context.broadcast(model.order)
            parameters['method'] = context.broadcast(type(model))
            parameters['lags'] = context.broadcast(model.lags)
            parameters['max_lag'] = context.broadcast(model.max_lag)

            func = lambda x: slave_train_univariate(x, **parameters)

            flrgs = context.parallelize(data).repartition(nodes*4).mapPartitions(func)

            for k in flrgs.collect():
                model.append_rule(k[1])

            return model
        else:
            if model.is_clustered:
                parameters['type'] = context.broadcast('clustered')
                names = []
                for name, fset in model.partitioner.sets.items():
                    names.append(name)
                    parameters['partitioner_{}'.format(name)] = context.broadcast([(k,v) for k,v in fset.sets.items()])

                parameters['partitioner_names'] = context.broadcast(names)

            else:
                parameters['type'] = context.broadcast('multivariate')
            names = []
            for var in model.explanatory_variables:
                #if var.data_type is None:
                #    raise Exception("It is mandatory to inform the data_type parameter for each variable when the training is distributed! ")
                names.append(var.name)
                parameters['{}_type'.format(var.name)] = context.broadcast(var.type)
                #parameters['{}_data_type'.format(var.name)] = context.broadcast(var.data_type)
                #parameters['{}_mask'.format(var.name)] = context.broadcast(var.mask)
                parameters['{}_label'.format(var.name)] = context.broadcast(var.data_label)
                parameters['{}_alpha'.format(var.name)] = context.broadcast(var.alpha_cut)
                parameters['{}_partitioner'.format(var.name)] = context.broadcast(var.partitioner.sets)
                parameters['{}_partitioner_type'.format(var.name)] = context.broadcast(var.partitioner.type)

            parameters['variables'] = context.broadcast(names)
            parameters['target'] = context.broadcast(model.target_variable.name)

            parameters['columns'] = context.broadcast(data.columns.values)

            data = data.to_dict(orient='records')

            parameters['alpha_cut'] = context.broadcast(model.alpha_cut)
            parameters['order'] = context.broadcast(model.order)
            parameters['method'] = context.broadcast(type(model))
            parameters['lags'] = context.broadcast(model.lags)
            parameters['max_lag'] = context.broadcast(model.max_lag)

            func = lambda x: slave_train_multivariate(x, **parameters)

            flrgs = context.parallelize(data).mapPartitions(func)

            for k in flrgs.collect():
                print(k)
                #for g in k:
                #    print(g)

                #return
                if parameters['type'].value == 'clustered':
                    if k[0] == 'counts':
                        for fset, count in k[1]:
                            model.partitioner.count[fset] = count
                    elif k[0] == 'flrgs':
                        model.append_rule(k[1])
                else:
                    model.append_rule(k[1])

            return model


def distributed_predict(data, model, url=SPARK_ADDR, app='pyFTS'):
    return None
