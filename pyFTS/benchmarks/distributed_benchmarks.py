"""
dispy Distributed Benchmarks to FTS methods

To enable a dispy cluster node:

python3 /usr/local/bin/dispynode.py -i [local IP] -d
"""

import datetime
import time

import dispy
import dispy.httpd
import numpy as np

from pyFTS.benchmarks import benchmarks, Util as bUtil, quantreg, arima
from pyFTS.common import Util
from pyFTS.partitioners import Grid


def run_point(mfts, partitioner, train_data, test_data, window_key=None, transformation=None, indexer=None):
    """
    Point forecast benchmark function to be executed on cluster nodes
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """
    import time
    from pyFTS.models import yu, chen, hofts, pwfts,ismailefendi,sadaei, song, cheng, hwang
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures, naive, arima, quantreg
    from pyFTS.common import Transformations

    tmp = [song.ConventionalFTS, chen.ConventionalFTS, yu.WeightedFTS, ismailefendi.ImprovedWeightedFTS,
           cheng.TrendWeightedFTS, sadaei.ExponentialyWeightedFTS, hofts.HighOrderFTS, hwang.HighOrderFTS,
           pwfts.ProbabilisticWeightedFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp4 = [naive.Naive, arima.ARIMA, quantreg.QuantileRegression]

    tmp3 = [Measures.get_point_statistics]

    tmp5 = [Transformations.Differential]

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "")
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner

    if transformation is not None:
        mfts.append_transformation(transformation)

    _start = time.time()
    mfts.train(train_data, partitioner.sets, order=mfts.order)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    _rmse, _smape, _u = Measures.get_point_statistics(test_data, mfts, indexer)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'rmse': _rmse, 'smape': _smape, 'u': _u, 'time': times, 'window': window_key}

    return ret


def point_sliding_window(data, windowsize, train=0.8, inc=0.1, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         benchmark_models=None, benchmark_models_parameters = None,
                         save=False, file=None, sintetic=False,nodes=None, depends=None):
    """
    Distributed sliding window benchmarks for FTS point forecasters
    :param data: 
    :param windowsize: size of sliding window
    :param train: percentual of sliding window data used to train the models
    :param inc: percentual of window is used do increment 
    :param models: FTS point forecasters
    :param partitioners: Universe of Discourse partitioner
    :param partitions: the max number of partitions on the Universe of Discourse 
    :param max_order: the max order of the models (for high order models)
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :param dump: 
    :param benchmark_models: Non FTS models to benchmark
    :param benchmark_models_parameters: Non FTS models parameters
    :param save: save results
    :param file: file path to save the results
    :param sintetic: if true only the average and standard deviation of the results
    :param nodes: list of cluster nodes to distribute tasks
    :param depends: list of module dependencies 
    :return: DataFrame with the results
    """

    cluster = dispy.JobCluster(run_point, nodes=nodes) #, depends=dependencies)

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))


    jobs = []
    objs = {}
    rmse = {}
    smape = {}
    u = {}
    times = {}

    pool = build_model_pool_point(models, max_order, benchmark_models, benchmark_models_parameters)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train, inc):
        experiments += 1

        benchmarks_only = {}

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                for _id, m in enumerate(pool,start=0):
                    if m.benchmark_only and m.shortname in benchmarks_only:
                        continue
                    else:
                        benchmarks_only[m.shortname] = m
                    job = cluster.submit(m, data_train_fs, train, test, ct, transformation)
                    job.id = _id  # associate an ID to identify jobs (if needed later)
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
            rmse[tmp['key']].append_rhs(tmp['rmse'])
            smape[tmp['key']].append_rhs(tmp['smape'])
            u[tmp['key']].append_rhs(tmp['u'])
            times[tmp['key']].append_rhs(tmp['time'])
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


def build_model_pool_point(models, max_order, benchmark_models, benchmark_models_parameters):
    pool = []

    if benchmark_models is None and models is None:
        benchmark_models = [arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA,
                            quantreg.QuantileRegression, quantreg.QuantileRegression]

    if benchmark_models_parameters is None:
        benchmark_models_parameters = [(1, 0, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2), 1, 2]

    if models is None:
        models = benchmarks.get_point_methods()
    for model in models:
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            mfts.order = 1
            pool.append(mfts)

    if benchmark_models is not None:
        for count, model in enumerate(benchmark_models, start=0):
            par = benchmark_models_parameters[count]
            mfts = model(str(par if par is not None else ""))
            mfts.order = par
            pool.append(mfts)
    return pool


def run_interval(mfts, partitioner, train_data, test_data, window_key=None, transformation=None, indexer=None):
    """
    Interval forecast benchmark function to be executed on cluster nodes
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """
    import time
    from pyFTS.models import hofts,ifts,pwfts
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures, arima, quantreg

    tmp = [hofts.HighOrderFTS, ifts.IntervalFTS,  pwfts.ProbabilisticWeightedFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp4 = [arima.ARIMA, quantreg.QuantileRegression]

    tmp3 = [Measures.get_interval_statistics]

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "") + str(mfts.alpha)
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner

    if transformation is not None:
        mfts.append_transformation(transformation)

    _start = time.time()
    mfts.train(train_data, partitioner.sets, order=mfts.order)
    _end = time.time()
    times = _end - _start

    _start = time.time()
    _sharp, _res, _cov, _q05, _q25, _q75, _q95 = Measures.get_interval_statistics(test_data, mfts)
    _end = time.time()
    times += _end - _start

    ret = {'key': _key, 'obj': mfts, 'sharpness': _sharp, 'resolution': _res, 'coverage': _cov, 'time': times,
           'Q05': _q05, 'Q25': _q25, 'Q75': _q75, 'Q95': _q95, 'window': window_key}

    return ret


def interval_sliding_window(data, windowsize, train=0.8,  inc=0.1, models=None, partitioners=[Grid.GridPartitioner],
                            partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                            benchmark_models=None, benchmark_models_parameters = None,
                            save=False, file=None, sintetic=False,nodes=None, depends=None):
    """
     Distributed sliding window benchmarks for FTS point_to_interval forecasters
     :param data: 
     :param windowsize: size of sliding window
     :param train: percentual of sliding window data used to train the models
     :param inc:
     :param models: FTS point forecasters
     :param partitioners: Universe of Discourse partitioner
     :param partitions: the max number of partitions on the Universe of Discourse 
     :param max_order: the max order of the models (for high order models)
     :param transformation: data transformation
     :param indexer: seasonal indexer
     :param dump: 
     :param benchmark_models:
     :param benchmark_models_parameters:
     :param save: save results
     :param file: file path to save the results
     :param sintetic: if true only the average and standard deviation of the results
     :param nodes: list of cluster nodes to distribute tasks
     :param depends: list of module dependencies 
     :return: DataFrame with the results
     """

    alphas = [0.05, 0.25]

    if benchmark_models is None and models is None:
        benchmark_models = [arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA,
                            quantreg.QuantileRegression, quantreg.QuantileRegression]

    if benchmark_models_parameters is None:
        benchmark_models_parameters = [(1, 0, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2), 1, 2]

    cluster = dispy.JobCluster(run_interval, nodes=nodes) #, depends=dependencies)

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    pool = []
    jobs = []
    objs = {}
    sharpness = {}
    resolution = {}
    coverage = {}
    q05 = {}
    q25 = {}
    q75 = {}
    q95 = {}
    times = {}

    if models is None:
        models = benchmarks.get_interval_methods()

    for model in models:
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            mfts.order = 1
            pool.append(mfts)

    if benchmark_models is not None:
        for count, model in enumerate(benchmark_models, start=0):
            for a in alphas:
                par = benchmark_models_parameters[count]
                mfts = model(str(par if par is not None else ""), alpha=a)
                mfts.order = par
                pool.append(mfts)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train, inc=inc):
        experiments += 1

        benchmarks_only = {}

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                for id, m in enumerate(pool,start=0):
                    if m.benchmark_only and m.shortname in benchmarks_only:
                        continue
                    else:
                        benchmarks_only[m.shortname] = m
                    job = cluster.submit(m, data_train_fs, train, test, ct, transformation)
                    job.id = id  # associate an ID to identify jobs (if needed later)
                    jobs.append(job)

    for job in jobs:
        tmp = job()
        if job.status == dispy.DispyJob.Finished and tmp is not None:
            if tmp['key'] not in objs:
                objs[tmp['key']] = tmp['obj']
                sharpness[tmp['key']] = []
                resolution[tmp['key']] = []
                coverage[tmp['key']] = []
                times[tmp['key']] = []
                q05[tmp['key']] = []
                q25[tmp['key']] = []
                q75[tmp['key']] = []
                q95[tmp['key']] = []

            sharpness[tmp['key']].append_rhs(tmp['sharpness'])
            resolution[tmp['key']].append_rhs(tmp['resolution'])
            coverage[tmp['key']].append_rhs(tmp['coverage'])
            times[tmp['key']].append_rhs(tmp['time'])
            q05[tmp['key']].append_rhs(tmp['Q05'])
            q25[tmp['key']].append_rhs(tmp['Q25'])
            q75[tmp['key']].append_rhs(tmp['Q75'])
            q95[tmp['key']].append_rhs(tmp['Q95'])
            print(tmp['key'])
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

    return bUtil.save_dataframe_interval(coverage, experiments, file, objs, resolution, save, sharpness, sintetic,
                                         times, q05, q25, q75, q95)


def run_ahead(mfts, partitioner, train_data, test_data, steps, resolution, window_key=None, transformation=None, indexer=None):
    """
    Probabilistic m-step ahead forecast benchmark function to be executed on cluster nodes
    :param mfts: FTS model
    :param partitioner: Universe of Discourse partitioner
    :param train_data: data used to train the model
    :param test_data: ata used to test the model 
    :param steps: 
    :param resolution: 
    :param window_key: id of the sliding window
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :return: a dictionary with the benchmark results 
    """
    import time
    import numpy as np
    from pyFTS.models import hofts, ifts, pwfts
    from pyFTS.models.ensemble import ensemble
    from pyFTS.partitioners import Grid, Entropy, FCM
    from pyFTS.benchmarks import Measures, arima
    from pyFTS.models.seasonal import SeasonalIndexer

    tmp = [hofts.HighOrderFTS, ifts.IntervalFTS, pwfts.ProbabilisticWeightedFTS, arima.ARIMA, ensemble.AllMethodEnsembleFTS]

    tmp2 = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner]

    tmp3 = [Measures.get_distribution_statistics, SeasonalIndexer.SeasonalIndexer, SeasonalIndexer.LinearSeasonalIndexer]

    if mfts.benchmark_only:
        _key = mfts.shortname + str(mfts.order if mfts.order is not None else "") + str(mfts.alpha)
    else:
        pttr = str(partitioner.__module__).split('.')[-1]
        _key = mfts.shortname + " n = " + str(mfts.order) + " " + pttr + " q = " + str(partitioner.partitions)
        mfts.partitioner = partitioner

    if transformation is not None:
        mfts.append_transformation(transformation)

    if mfts.has_seasonality:
        mfts.indexer = indexer

    try:
        _start = time.time()
        mfts.train(train_data, partitioner.sets, order=mfts.order)
        _end = time.time()
        times = _end - _start

        _crps1, _crps2, _t1, _t2 = Measures.get_distribution_statistics(test_data, mfts, steps=steps,
                                                              resolution=resolution)
        _t1 += times
        _t2 += times
    except Exception as e:
        print(e)
        _crps1 = np.nan
        _crps2 = np.nan
        _t1 = np.nan
        _t2 = np.nan

    ret = {'key': _key, 'obj': mfts, 'CRPS_Interval': _crps1, 'CRPS_Distribution': _crps2, 'TIME_Interval': _t1,
           'TIME_Distribution': _t2, 'window': window_key}

    return ret


def ahead_sliding_window(data, windowsize, steps, resolution, train=0.8, inc=0.1, models=None, partitioners=[Grid.GridPartitioner],
                         partitions=[10], max_order=3, transformation=None, indexer=None, dump=False,
                         benchmark_models=None, benchmark_models_parameters = None,
                         save=False, file=None, synthetic=False, nodes=None):
    """
    Distributed sliding window benchmarks for FTS probabilistic forecasters
    :param data: 
    :param windowsize: size of sliding window
    :param train: percentual of sliding window data used to train the models
    :param steps: 
    :param resolution: 
    :param models: FTS point forecasters
    :param partitioners: Universe of Discourse partitioner
    :param partitions: the max number of partitions on the Universe of Discourse 
    :param max_order: the max order of the models (for high order models)
    :param transformation: data transformation
    :param indexer: seasonal indexer
    :param dump: 
    :param save: save results
    :param file: file path to save the results
    :param synthetic: if true only the average and standard deviation of the results
    :param nodes: list of cluster nodes to distribute tasks
    :param depends: list of module dependencies 
    :return: DataFrame with the results 
    """

    alphas = [0.05, 0.25]

    if benchmark_models is None and models is None:
        benchmark_models = [arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA]

    if benchmark_models_parameters is None:
        benchmark_models_parameters = [(1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 0, 1), (2, 0, 2)]

    cluster = dispy.JobCluster(run_ahead, nodes=nodes)  # , depends=dependencies)

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    _process_start = time.time()

    print("Process Start: {0: %H:%M:%S}".format(datetime.datetime.now()))

    pool = []
    jobs = []
    objs = {}
    crps_interval = {}
    crps_distr = {}
    times1 = {}
    times2 = {}

    if models is None:
        models = benchmarks.get_probabilistic_methods()

    for model in models:
        mfts = model("")

        if mfts.is_high_order:
            for order in np.arange(1, max_order + 1):
                if order >= mfts.min_order:
                    mfts = model("")
                    mfts.order = order
                    pool.append(mfts)
        else:
            pool.append(mfts)

    if benchmark_models is not None:
        for count, model in enumerate(benchmark_models, start=0):
            for a in alphas:
                par = benchmark_models_parameters[count]
                mfts = model(str(par if par is not None else ""), alpha=a, dist=True)
                mfts.order = par
                pool.append(mfts)

    experiments = 0
    for ct, train, test in Util.sliding_window(data, windowsize, train, inc=inc):
        experiments += 1

        benchmarks_only = {}

        if dump: print('\nWindow: {0}\n'.format(ct))

        for partition in partitions:

            for partitioner in partitioners:

                data_train_fs = partitioner(train, partition, transformation=transformation)

                for id, m in enumerate(pool,start=0):
                    if m.benchmark_only and m.shortname in benchmarks_only:
                        continue
                    else:
                        benchmarks_only[m.shortname] = m
                    job = cluster.submit(m, data_train_fs, train, test, steps, resolution, ct, transformation, indexer)
                    job.id = id  # associate an ID to identify jobs (if needed later)
                    jobs.append(job)

    for job in jobs:
        tmp = job()
        if job.status == dispy.DispyJob.Finished and tmp is not None:
            if tmp['key'] not in objs:
                objs[tmp['key']] = tmp['obj']
                crps_interval[tmp['key']] = []
                crps_distr[tmp['key']] = []
                times1[tmp['key']] = []
                times2[tmp['key']] = []
            crps_interval[tmp['key']].append_rhs(tmp['CRPS_Interval'])
            crps_distr[tmp['key']].append_rhs(tmp['CRPS_Distribution'])
            times1[tmp['key']].append_rhs(tmp['TIME_Interval'])
            times2[tmp['key']].append_rhs(tmp['TIME_Distribution'])

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

    return bUtil.save_dataframe_ahead(experiments, file, objs, crps_interval, crps_distr, times1, times2, save, synthetic)
