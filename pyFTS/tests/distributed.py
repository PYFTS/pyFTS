from pyFTS.partitioners import Grid
from pyFTS.models import chen
from pyFTS.benchmarks import Measures
from pyFTS.common import Util as cUtil, fts
import pandas as pd
import numpy as np
import os
from pyFTS.common import Transformations
from copy import deepcopy
from pyFTS.nonstationary import flrg, util, honsfts, partitioners
from pyFTS.models.nonstationary import nsfts

bc = Transformations.BoxCox(0)

import dispy
import dispy.httpd

os.chdir("/home/petronio/Dropbox/Doutorado/Codigos/")


def evaluate_individual_model(model, partitioner, train, test, window_size, time_displacement):
    import numpy as np
    from pyFTS.partitioners import Grid
    from pyFTS.benchmarks import Measures

    try:
        model.train(train, sets=partitioner.sets, order=model.order, parameters=window_size)
        forecasts = model.forecast(test, time_displacement=time_displacement, window_size=window_size)
        _rmse = Measures.rmse(test[model.order:], forecasts[:-1])
        _mape = Measures.mape(test[model.order:], forecasts[:-1])
        _u = Measures.UStatistic(test[model.order:], forecasts[:-1])
    except Exception as e:
        print(e)
        _rmse = np.nan
        _mape = np.nan
        _u = np.nan

    return {'model': model.shortname, 'partitions': partitioner.partitions, 'order': model.order,
            'rmse': _rmse, 'mape': _mape, 'u': _u}


data = pd.read_csv("DataSets/synthetic_nonstationary_dataset_A.csv", sep=";")
data = np.array(data["0"][:])

cluster = dispy.JobCluster(evaluate_individual_model, nodes=['192.168.0.108', '192.168.0.110'])
http_server = dispy.httpd.DispyHTTPServer(cluster)

jobs = []

models = []

for order in [1, 2, 3]:
    if order == 1:
        model = nsfts.NonStationaryFTS("")
    else:
        model = honsfts.HighOrderNonStationaryFTS("")

    model.order = order

    models.append(model)

for ct, train, test in cUtil.sliding_window(data, 300):
    for partition in np.arange(5, 100, 1):
        tmp_partitioner = Grid.GridPartitioner(train, partition)
        partitioner = partitioners.PolynomialNonStationaryPartitioner(train, tmp_partitioner,
                                                                      window_size=35, degree=1)
        for model in models:
            # print(model.shortname, partition, model.order)
            #job = evaluate_individual_model(model, train, test)
            job = cluster.submit(deepcopy(model), deepcopy(partitioner), train, test, 35, 240)
            job.id = ct + model.order*100
            jobs.append(job)

results = {}

for job in jobs:
    tmp = job()
    # print(tmp)
    if job.status == dispy.DispyJob.Finished and tmp is not None:
        _m = tmp['model']
        _o = tmp['order']
        _p = tmp['partitions']
        if _m not in results:
            results[_m] = {}
        if _o not in results[_m]:
            results[_m][_o] = {}
        if _p not in results[_m][_o]:
            results[_m][_o][_p] = {}
            results[_m][_o][_p]['rmse'] = []
            results[_m][_o][_p]['mape'] = []
            results[_m][_o][_p]['u'] = []

        results[_m][_o][_p]['rmse'].append_rhs(tmp['rmse'])
        results[_m][_o][_p]['mape'].append_rhs(tmp['mape'])
        results[_m][_o][_p]['u'].append_rhs(tmp['u'])

cluster.wait()  # wait for all jobs to finish

cluster.print_status()

http_server.shutdown()  # this waits until browser gets all updates
cluster.close()

dados = []
ncolunas = None

for m in sorted(results.keys()):
    for o in sorted(results[m].keys()):
        for p in sorted(results[m][o].keys()):
            for r in ['rmse', 'mape', 'u']:
                tmp = []
                tmp.append(m)
                tmp.append(o)
                tmp.append(p)
                tmp.append(r)
                tmp.extend(results[m][o][p][r])

                dados.append(tmp)

                if ncolunas is None:
                    ncolunas = len(results[m][o][p][r])

colunas = ["model", "order", "partitions",  "metric"]
for k in np.arange(0, ncolunas):
    colunas.append(str(k))

dat = pd.DataFrame(dados, columns=colunas)
dat.to_csv("experiments/nsfts_partitioning_A.csv", sep=";")
