import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyFTS.common import FuzzySet,SortedCollection,tree
from pyFTS.probabilistic import ProbabilityDistribution


class Mixture(ProbabilityDistribution.ProbabilityDistribution):
    """

    """
    def __init__(self, type="mixture", **kwargs):
        self.models = []
        self.weights = []

    def append_model(self,model, weight):
        self.models.append(model)
        self.weights.append(weight)

    def density(self, values):
        if not isinstance(values, list):
            values = [values]

        for ct, m in enumerate(self.models):

        probs = [m.density(values) ]

