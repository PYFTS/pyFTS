"""
Chiu, Stephen L. "Fuzzy model identification based on cluster estimation." Journal of Intelligent & fuzzy systems 2.3 (1994): 267-278.
"""

import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


def imax(vec):
    i = np.argmax(vec)
    return (i, vec[i])


def subclust(data, ra, rb, eps_sup, eps_inf):
    if len(data.shape) == 1:
        data = np.reshape(data, (data.shape[0], 1))

    centers = np.zeros((0, data.shape[1]))

    # Initial potentials
    alpha = 4/ra**2
    beta = 4/rb**2

    pot = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        pot[i] = np.sum(np.exp(-alpha*np.linalg.norm(data - data[i,:], axis=1)**2))

    pot_max_i, pot_max = imax(pot)

    current_pot_max = pot_max
    while current_pot_max:
        x_star = data[pot_max_i,:]
        accept = False
        if current_pot_max > eps_sup * pot_max:
            # Accept xk as a cluster center and continue
            accept = True
        elif current_pot_max <= eps_inf * pot_max:
            # Reject xk and end the clustering process
            break
        else:
            d_min = np.min(np.linalg.norm(x_star - centers))
            if d_min/ra + current_pot_max/pot_max >= 1:
                accept = True
            else:
                pot[pot_max_i] = 0
                pot_max_i, current_pot_max = imax(pot)
        if accept:
            centers = np.vstack((centers, x_star))
            # Recompute potentials
            for i in range(data.shape[0]):
                new_pot = pot[i] - current_pot_max*np.exp(-beta*np.linalg.norm(x_star - data[i,:])**2)
                new_pot = max(0, new_pot)
                pot[i] = new_pot
            pot_max_i, current_pot_max = imax(pot)
    return centers


class SubClustPartitioner(partitioner.Partitioner):
    """Subtractive Clustering Partitioner"""
    def __init__(self, **kwargs):
        self.ra = kwargs.get('ra', 0.8)
        self.rb = kwargs.get('rb', self.ra * 1.5)
        self.eps_sup = kwargs.get('eps_sup', 0.5)
        self.eps_inf = kwargs.get('eps_inf', 0.15)
        super(SubClustPartitioner, self).__init__(name="SubClust", **kwargs)

    def build(self, data):
        sets = {}

        kwargs = {'type': self.type, 'variable': self.variable}

        partitions = subclust(data, self.ra, self.rb, self.epssup, self.epsinf)
        partitions = list(np.reshape(partitions, partitions.shape[0]))
        partitions.append(self.min)
        partitions.append(self.max)
        partitions = list(set(partitions))
        partitions.sort()
        self.partitions = len(partitions)
        for c in np.arange(1, len(partitions)-1):
            _name = self.get_name(c-1)
            if self.membership_function == Membership.trimf:
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                              [partitions[c - 1], partitions[c], partitions[c + 1]],partitions[c], **kwargs)
            elif self.membership_function == Membership.trapmf:
                b1 = (partitions[c] - partitions[c - 1])/2
                b2 = (partitions[c + 1] - partitions[c]) / 2
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trapmf,
                                              [partitions[c - 1], partitions[c] - b1,
                                               partitions[c] + b2, partitions[c + 1]],
                                              partitions[c], **kwargs)

        return sets
