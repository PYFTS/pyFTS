from pyFTS.common import Membership, FuzzySet as FS
from pyFTS.common.Composite import FuzzySet as Composite
from pyFTS.partitioners import partitioner, Grid
from pyFTS.models.seasonal.common import DateTime, FuzzySet, strip_datepart
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import KDTree


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
        '''Seasonality, a pyFTS.models.seasonal.common.DateTime object'''
        self.mask = kwargs.get('mask', '%Y-%m-%d %H:%M:%S')
        '''A string with datetime formating mask'''

        data = kwargs.get('data', None)
        if self.season == DateTime.year:
            ndata = [strip_datepart(k, self.season) for k in data]
            self.min = min(ndata)
            self.max = max(ndata)
        else:
            tmp = (self.season.value / self.partitions) / 2
            self.min = tmp
            self.max = self.season.value + tmp

        self.type = kwargs.get('type','seasonal')

        self.sets = self.build(None)

        if self.ordered_sets is None and self.setnames is not None:
            self.ordered_sets = self.setnames
        else:
            self.ordered_sets = FS.set_ordered(self.sets)

    def extractor(self,x):
        if self.type == 'seasonal':
            return strip_datepart(x, self.season, self.mask)
        else:
            return x

    def build(self, data):
        sets = {}

        kwargs = {'variable': self.variable, 'type': self.type }

        if self.season == DateTime.year:
            dlen = (self.max - self.min)
            partlen = dlen / self.partitions
        elif self.season == DateTime.day_of_week:
            self.min, self.max, partlen, pl2 = 0, 7, 1, 1
        elif self.season == DateTime.minute:
            self.min, self.max, partlen, pl2 = 0, 60, 1, 1
        elif self.season == DateTime.hour:
            self.min, self.max, partlen, pl2 = 0, 24, 1, 1
        elif self.season == DateTime.month:
            self.min, self.max, partlen, pl2 = 1, 13, 1, 1
        elif self.season  in (DateTime.half, DateTime.third, DateTime.quarter, DateTime.sixth):
            self.min, self.max, partlen, pl2 = 1, self.season.value+1, 1, 1
        else:
            partlen = self.season.value / self.partitions
            pl2 = partlen / 2

        count = 0
        for c in np.arange(self.min, self.max, partlen):
            set_name = self.get_name(count)
            if self.membership_function == Membership.trimf:
                if c == self.min:
                    tmp = Composite(set_name, superset=True, **kwargs)
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trimf,
                                            [self.season.value - pl2, self.season.value,
                                             self.season.value + pl2], self.season.value, alpha=1,
                                            **kwargs))
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trimf,
                                            [c - partlen, c, c + partlen], c,
                                            **kwargs))
                    tmp.centroid = c
                    sets[set_name] = tmp
                elif c == self.max - partlen:
                    tmp = Composite(set_name, superset=True, **kwargs)
                    tmp.append_set(FuzzySet(self.season, set_name, Membership.trimf,
                                            [-pl2, 0.0,
                                             pl2], 0.0, alpha=1,
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

    def build_index(self):
        points = []

        fset = self.sets[self.ordered_sets[0]]
        points.append([fset.sets[1].lower, fset.sets[1].centroid, fset.sets[1].upper])

        for ct, key in enumerate(self.ordered_sets[1:-1]):
            fset = self.sets[key]
            points.append([fset.lower, fset.centroid, fset.upper])

        fset = self.sets[self.ordered_sets[-1]]
        points.append([fset.sets[1].lower, fset.sets[1].centroid, fset.sets[1].upper])

        import sys
        sys.setrecursionlimit(100000)

        self.kdtree = KDTree(points)

        sys.setrecursionlimit(1000)

    def search(self, data, **kwargs):
        '''
        Perform a search for the nearest fuzzy sets of the point 'data'. This function were designed to work with several
        overlapped fuzzy sets.

        :param data: the value to search for the nearest fuzzy sets
        :param type: the return type: 'index' for the fuzzy set indexes or 'name' for fuzzy set names.
        :param results: the number of nearest fuzzy sets to return
        :return: a list with the nearest fuzzy sets
        '''

        type = kwargs.get('type','index')
        results = kwargs.get('results',3)

        if self.kdtree is None:
            self.build_index()

        _, ix = self.kdtree.query([data, data, data], results)

        ix = ix.tolist()

        if 0 in ix:
            ix.insert(0, self.partitions-1)
        elif self.partitions-1 in ix:
            ix.insert(0, 0)

        if type == 'name':
            return [self.ordered_sets[k] for k in sorted(ix)]
        else:
            return sorted(ix)


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
