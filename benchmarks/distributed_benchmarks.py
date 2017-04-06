import random
import dispy
import dispy.httpd
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import datetime
import pyFTS
from pyFTS.partitioners import partitioner, Grid, Huarng, Entropy, FCM
from pyFTS.benchmarks import Measures, naive, arima, ResidualAnalysis, ProbabilityDistribution
from pyFTS.common import Membership, FuzzySet, FLR, Transformations, Util
from pyFTS import fts, chen, yu, ismailefendi, sadaei, hofts, hwang,  pwfts, ifts
from pyFTS.benchmarks import  benchmarks, parallel_benchmarks, Util as bUtil


def run_point(mfts, partitioner, train_data, test_data, window_key=None, transformation=None, indexer=None):
    import time
    from pyFTS import yu,chen,hofts,ifts,pwfts,ismailefendi,sadaei
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures

    tmp = [yu.WeightedFTS, chen.ConventionalFTS, hofts.HighOrderFTS, ifts.IntervalFTS,
           pwfts.ProbabilisticWeightedFTS, ismailefendi.ImprovedWeightedFTS, sadaei.ExponentialyWeightedFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp3 = [Measures.get_point_statistics]

    pttr = str(partitioner.__module__).split('.')[-1]
    _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
    mfts.partitioner = partitioner
    if transformation is not None:
        mfts.appendTransformation(transformation)

#    try:
    _start = time.time()
    mfts.train(train_data, partitioner.sets, order=mfts.order)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    _rmse, _smape, _u = Measures.get_point_statistics(test_data, mfts, indexer)
    _end = time.time()
    times += _end - _start
 #   except Exception as e:
 #       print(e)
 #       _rmse = np.nan
 #       _smape = np.nan
 #       _u = np.nan
 #       times = np.nan

    ret = {'key': _key, 'obj': mfts, 'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times, 'window': window_key}

 #   print(ret)

    return ret


def point_sliding_window(data, windowsize, train=0.8, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         save=False, file=None, sintetic=False,nodes=None, depends=None):

#    dependencies = [fts, Membership, benchmarks]
#    if depends is not None: dependencies.extend(depends)

#    if models is not None:
#        dependencies.extend(models)
#    else:
#        dependencies.extend(benchmarks.get_point_methods())

#    dependencies.extend(partitioners)
#    if transformation is not None: dependencies.extend(transformation)
#    if indexer is not None: dependencies.extend(indexer)

    cluster = dispy.JobCluster(run_point, nodes=nodes) #, depends=dependencies)

    # import dispy's httpd module, create http server for this cluster

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    pool = []
    jobs = []
    objs = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    if models is None:
        models = benchmarks.get_point_methods()

    for model in models:
        mfts = model("")

        if mfts.isHighOrder:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.minOrder:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            pool.append(mfts)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train):
        experiments += 1

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                for id, m in enumerate(pool,start=0):
                    job = cluster.submit(m, data_train_fs, train, test, ct, transformation)
                    job.id = id  # associate an ID to identify jobs (if needed later)
                    jobs.append(job)

    for job in jobs:
        tmp = job()
        if job.status == dispy.DispyJob.Finished and tmp is not None:
            if tmp['key'] not in objs:
                objs[tmp['key']] = tmp['obj']
                rmse[tmp['key']] = []
                smape[tmp['key']] = []
                u[tmp['key']] = []
                times[tmp['key']] = []
            rmse[tmp['key']].append(tmp['rmse'])
            smape[tmp['key']].append(tmp['smape'])
            u[tmp['key']].append(tmp['u'])
            times[tmp['key']].append(tmp['time'])
            print(tmp['key'], tmp['window'])
        else:
            print(job.exception)
            print(job.stdout)

    _process_end = time.time()

    print("Process End: {0: %H:%M:%S}".format(datetime.datetime.now()))

    print("Process Duration: {0}".format(_process_end - _process_start))

    cluster.wait()  # wait for all jobs to finish

    cluster.print_status()

    http_server.shutdown()  # this waits until browser gets all updates
    cluster.close()

    return bUtil.save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u)


def compute(data):
    import socket
    return (socket.gethostname(), data)


def teste(data,nodes):
    cluster = dispy.JobCluster(compute, nodes=nodes)
    jobs = []
    for ct, train, test in Util.sliding_window(data, 2000, 0.8):
        job = cluster.submit(ct)
        jobs.append(job)

    for job in jobs:
        x = job()
        print(x)

    cluster.wait()
    cluster.close()


