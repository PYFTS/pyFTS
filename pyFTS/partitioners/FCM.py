"""
S. T. Li, Y. C. Cheng, and S. Y. Lin, “A FCM-based deterministic forecasting model for fuzzy time series,”
Comput. Math. Appl., vol. 56, no. 12, pp. 3052–3063, Dec. 2008. DOI: 10.1016/j.camwa.2008.07.033.
"""
import numpy as np
import pandas as pd
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


def fuzzy_distance(x, y):
    if isinstance(x, (list, tuple, np.ndarray)):
        tmp = functools.reduce(operator.add, [(x[k] - y[k]) ** 2 for k in range(0, len(x))])
    else:
        tmp = (x - y) ** 2
    return math.sqrt(tmp)


def membership(val, vals):
    soma = 0
    for k in vals:
        if k == 0:
            k = 1
        soma = soma + (val / k) ** 2

    return soma


def fuzzy_cmeans(k, data, size, m, deltadist=0.001):
    data_length = len(data)

    # Centroid initialization
    centroids = [data[rnd.randint(0, data_length - 1)] for kk in range(0, k)]

    # Membership table
    membership_table = np.zeros((k, data_length)) #[[0 for kk in range(0, k)] for xx in range(0, data_length)]

    mean_change = 1000

    m_exp = 1 / (m - 1)

    iterations = 0

    while iterations < 1000 and mean_change > deltadist:

        mean_change = 0
        inst_count = 0
        for instance in data:

            dist_groups = np.zeros(k) #[0 for xx in range(0, k)]

            for group_count, group in enumerate(centroids):
                dist_groups[group_count] = fuzzy_distance(group, instance)

            dist_groups_total = functools.reduce(operator.add, [xk for xk in dist_groups])

            for grp in range(0, k):
                if dist_groups[grp] == 0:
                    membership_table[inst_count][grp] = 1
                else:
                    membership_table[inst_count][grp] = 1 / membership(dist_groups[grp], dist_groups)
                    # membership_table[inst_count][grp] = 1/(dist_groups[grp] / dist_grupos_total)
                    # membership_table[inst_count][grp] = (1/(dist_groups[grp]**2))**m_exp / (1/(dist_grupos_total**2))**m_exp

            inst_count = inst_count + 1

        for group_count, group in enumerate(centroids):
            if size > 1:
                oldgrp = [xx for xx in group]
                for atr in range(0, size):
                    soma = functools.reduce(operator.add,
                                            [membership_table[xk][group_count] * data[xk][atr] for xk in range(0, data_length)])
                    norm = functools.reduce(operator.add, [membership_table[xk][group_count] for xk in range(0, data_length)])
                    centroids[group_count][atr] = soma / norm
            else:
                oldgrp = group
                soma = functools.reduce(operator.add,
                                        [membership_table[xk][group_count] * data[xk] for xk in range(0, data_length)])
                norm = functools.reduce(operator.add, [membership_table[xk][group_count] for xk in range(0, data_length)])
                centroids[group_count] = soma / norm

            mean_change = mean_change + fuzzy_distance(oldgrp, group)

        mean_change = mean_change / k
        iterations = iterations + 1

    return centroids


class FCMPartitioner(partitioner.Partitioner):
    """
    
    """

    def __init__(self, **kwargs):
        super(FCMPartitioner, self).__init__(name="FCM", **kwargs)

    def build(self, data):
        sets = {}

        kwargs = {'type': self.type, 'variable': self.variable}

        centroids = fuzzy_cmeans(self.partitions, data, 1, 2)
        centroids.append(self.max)
        centroids.append(self.min)
        centroids = list(set(centroids))
        centroids.sort()
        for c in np.arange(1, len(centroids) - 1):
            _name = self.get_name(c)
            if self.membership_function == Membership.trimf:
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                                [round(centroids[c - 1], 3), round(centroids[c], 3),
                                                 round(centroids[c + 1], 3)],
                                                round(centroids[c], 3), **kwargs)
            elif self.membership_function == Membership.trapmf:
                q1 = (round(centroids[c], 3) - round(centroids[c - 1], 3)) / 2
                q2 = (round(centroids[c + 1], 3) - round(centroids[c], 3)) / 2
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                                [round(centroids[c - 1], 3), round(centroids[c], 3) - q1,
                                                 round(centroids[c], 3) + q2, round(centroids[c + 1], 3)],
                                                round(centroids[c], 3), **kwargs)

        return sets
