"""Class Partitioner with Singleton Fuzzy Sets"""

import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class ClassPartitioner(partitioner.Partitioner):
  """Class Partitioner: Given a dictionary with class/values pairs, create singleton fuzzy sets for each class"""

  def __init__(self, **kwargs):
    """
    Class Partitioner
    """
    super(ClassPartitioner, self).__init__(name="Class", preprocess = False)

    self.ordered_sets = []

    self.min = 0
    self.max = 0
    self.partitions = 0

    classes = kwargs.get("classes", {})

    for k,v in classes.items():
      self.min = min([self.min, v])
      self.max = max([self.max, v])
      self.partitions += 1
      self.sets[k] = FuzzySet.FuzzySet(k, Membership.singleton, [v], v, **kwargs)
      self.ordered_sets.append(k)

  def build(self, data : list):
    pass
