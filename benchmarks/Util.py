"""
Benchmark utility functions
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from pyFTS.common import Util


def save_dataframe_point(experiments, file, objs, rmse, save, sintetic, smape, times, u):
    """
    Create a dataframe to store the benchmark results
    :param experiments: dictionary with the execution results
    :param file: 
    :param objs: 
    :param rmse: 
    :param save: 
    :param sintetic: 
    :param smape: 
    :param times: 
    :param u: 
    :return: 
    """
    ret = []

    if sintetic:

        for k in sorted(objs.keys()):
            try:
                mod = []
                mfts = objs[k]
                mod.append(mfts.shortname)
                mod.append(mfts.order)
                mod.append(mfts.partitioner.name)
                mod.append(mfts.partitioner.partitions)
                mod.append(len(mfts))
                mod.append(np.round(np.nanmean(rmse[k]), 2))
                mod.append(np.round(np.nanstd(rmse[k]), 2))
                mod.append(np.round(np.nanmean(smape[k]), 2))
                mod.append(np.round(np.nanstd(smape[k]), 2))
                mod.append(np.round(np.nanmean(u[k]), 2))
                mod.append(np.round(np.nanstd(u[k]), 2))
                mod.append(np.round(np.nanmean(times[k]), 4))
                ret.append(mod)
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)

        columns = ["Model", "Order", "Scheme","Partitions", "Size", "RMSEAVG", "RMSESTD", "SMAPEAVG", "SMAPESTD", "UAVG", "USTD", "TIMEAVG"]
    else:
        for k in sorted(objs.keys()):
            try:
                mfts = objs[k]
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts), 'RMSE']
                tmp.extend(rmse[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),  'SMAPE']
                tmp.extend(smape[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),  'U']
                tmp.extend(u[k])
                ret.append(deepcopy(tmp))
                tmp = [mfts.shortname, mfts.order, mfts.partitioner.name, mfts.partitioner.partitions, len(mfts),  'TIME']
                tmp.extend(times[k])
                ret.append(deepcopy(tmp))
            except Exception as ex:
                print("Erro ao salvar ", k)
                print("Exceção ", ex)
        columns = [str(k) for k in np.arange(0, experiments)]
        columns.insert(0, "Model")
        columns.insert(1, "Order")
        columns.insert(2, "Scheme")
        columns.insert(3, "Partitions")
        columns.insert(4, "Size")
        columns.insert(5, "Measure")
    dat = pd.DataFrame(ret, columns=columns)
    if save: dat.to_csv(Util.uniquefilename(file), sep=";")
    return dat
