from pyFTS.common import Membership, FuzzySet as FS
from pyFTS.common.Composite import FuzzySet as Composite
from pyFTS.partitioners import partitioner, Grid
from pyFTS.models.seasonal.common import DateTime, FuzzySet, strip_datepart
import numpy as np
import matplotlib.pylab as plt


class TimeGridPartitioner(partitioner.Partitioner):
    """Even Length DateTime Grid Partitioner"""

    def __init__(self, **kwargs):
        """
        Even Length Grid Partitioner
        :param seasonality: Time granularity, from pyFTS.models.seasonal.common.DateTime
        :param data: Training data of which the universe of discourse will be extracted. The universe of discourse is the open interval between the minimum and maximum values of the training data.
        :param npart: The number of universe of discourse partitions, i.e., the number of fuzzy sets that will be created
        :param func: Fuzzy membership function (pyFTS.common.Membership)
        """
        super(TimeGridPartitioner, self).__init__(name="TimeGrid", preprocess=False, **kwargs)

        self.season = kwargs.get('seasonality', DateTime.day_of_year)
        data = kwargs.get('data', None)
        if self.season == DateTime.year:
            ndata = [strip_datepart(k, self.season) for k in data]
            self.min = min(ndata)
            self.max = max(ndata)
        else:
            tmp = (self.season.value / self.partitions) / 2
            self.min = tmp
            self.max = self.season.value + tmp

        self.sets = self.build(None)

        if self.ordered_sets is None and self.setnames is not None:
            self.ordered_sets = self.setnames
        else:
            self.ordered_sets = FS.set_ordered(self.sets)

    def build(self, data):
        sets = {}

        kwargs = {'variable': self.variable}

        if self.season == DateTime.year:
            dlen = (self.max - self.min)
            partlen = dlen / self.partitions
        else:
            partlen = self.season.value / self.partitions
            pl2 = partlen / 2

        count = 0
        for c in np.arange(self.min, self.max, partlen):
            set_name = self.get_name(count)
            if self.membership_function == Membership.trimf:
                if c == self.min:
                    tmp = Composite(set_name, superset=True)
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trimf,
                                            [self.season.value - pl2, self.season.value,
                                             self.season.value + 0.0000001], self.season.value, alpha=.5,
                                            **kwargs))
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trimf,
                                            [c - partlen, c, c + partlen], c,
                                            **kwargs))
                    tmp.centroid = c
                    sets[set_name] = tmp
                else:
                    sets[set_name] = FuzzySet(self.season, set_name, Membership.trimf,
                                         [c - partlen, c, c + partlen], c,
                                         **kwargs)
            elif self.membership_function == Membership.gaussmf:
                sets[set_name] = FuzzySet(self.season, set_name, Membership.gaussmf, [c, partlen / 3], c,
                                     **kwargs)
            elif self.membership_function == Membership.trapmf:
                q = partlen / 4
                if c == self.min:
                    tmp = Composite(set_name, superset=True)
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trimf,
                                            [self.season.value - pl2, self.season.value,
                                             self.season.value + 0.0000001], 0,
                                            **kwargs))
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trapmf,
                                            [c - partlen, c - q, c + q, c + partlen], c,
                                            **kwargs))
                    tmp.centroid = c
                    sets[set_name] = tmp
                else:
                    sets[set_name] = FuzzySet(self.season, set_name, Membership.trapmf,
                                         [c - partlen, c - q, c + q, c + partlen], c,
                                         **kwargs)
            count += 1

        self.min = 0

        return sets


    def plot(self, ax):
        """
        Plot the
        :param ax:
        :return:
        """
        ax.set_title(self.name)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, self.season.value])
        ticks = []
        x = []
        for key in self.sets.keys():
            s = self.sets[key]
            if s.type == 'composite':
                for ss in s.sets:
                    self.plot_set(ax, ss)
            else:
                self.plot_set(ax, s)
            ticks.append(str(round(s.centroid, 0)) + '\n' + s.name)
            x.append(s.centroid)
        ax.xaxis.set_ticklabels(ticks)
        ax.xaxis.set_ticks(x)
