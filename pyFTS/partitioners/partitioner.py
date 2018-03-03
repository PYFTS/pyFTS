from pyFTS.common import FuzzySet, Membership
import numpy as np
import matplotlib.pylab as plt


class Partitioner(object):
    """
    Universe of Discourse partitioner. Split data on several fuzzy sets
    """

    def __init__(self, **kwargs):
        """
        Universe of Discourse partitioner scheme. Split data on several fuzzy sets
        :param name: partitioner name
        :param data: Training data of which the universe of discourse will be extracted. The universe of discourse is the open interval between the minimum and maximum values of the training data.
        :param npart: The number of universe of discourse partitions, i.e., the number of fuzzy sets that will be created
        :param func: Fuzzy membership function (pyFTS.common.Membership)
        :param names: list of partitions names. If None is given the partitions will be auto named with prefix
        :param prefix: prefix of auto generated partition names
        :param transformation: data transformation to be applied on data
        """
        self.name = kwargs.get('name',"")
        self.partitions = kwargs.get('npart', 10)
        self.sets = {}
        self.membership_function = kwargs.get('func', Membership.trimf)
        self.setnames = kwargs.get('names', None)
        self.prefix = kwargs.get('prefix', 'A')
        self.transformation = kwargs.get('transformation', None)
        self.indexer = kwargs.get('indexer', None)
        self.variable = kwargs.get('variable', None)
        self.type = kwargs.get('type', 'common')
        self.ordered_sets = None

        if kwargs.get('preprocess',True):

            data = kwargs.get('data',[None])

            if self.indexer is not None:
                ndata = self.indexer.get_data(data)
            else:
                ndata = data

            if self.transformation is not None:
                ndata = self.transformation.apply(ndata)
            else:
                ndata = data

            _min = min(ndata)
            if _min < 0:
                self.min = _min * 1.1
            else:
                self.min = _min * 0.9

            _max = max(ndata)
            if _max > 0:
                self.max = _max * 1.1
            else:
                self.max = _max * 0.9

            self.sets = self.build(ndata)

            if self.ordered_sets is None and self.setnames is not None:
                self.ordered_sets = self.setnames
            else:
                self.ordered_sets = FuzzySet.set_ordered(self.sets)

            del(ndata)

    def build(self, data):
        """
        Perform the partitioning of the Universe of Discourse
        :param data: 
        :return: 
        """
        pass

    def get_name(self, counter):
        return self.prefix + str(counter) if self.setnames is None else self.setnames[counter]

    def plot(self, ax):
        """
        Plot the 
        :param ax: 
        :return: 
        """
        ax.set_title(self.name)
        ax.set_ylim([0, 1])
        ax.set_xlim([self.min, self.max])
        ticks = []
        x = []
        for key in self.sets.keys():
            s = self.sets[key]
            if s.type == 'common':
                self.plot_set(ax, s)
            elif s.type == 'composite':
                for ss in s.sets:
                    self.plot_set(ax, ss)
            ticks.append(str(round(s.centroid,0))+'\n'+s.name)
            x.append(s.centroid)
        ax.xaxis.set_ticklabels(ticks)
        ax.xaxis.set_ticks(x)

    def plot_set(self, ax, s):
        if s.mf == Membership.trimf:
            ax.plot([s.parameters[0], s.parameters[1], s.parameters[2]], [0, s.alpha, 0])
        elif s.mf == Membership.gaussmf:
            tmpx = [kk for kk in np.arange(s.lower, s.upper)]
            tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
            ax.plot(tmpx, tmpy)
        elif s.mf == Membership.trapmf:
            ax.plot(s.parameters, [0, s.alpha, s.alpha, 0])


    def __str__(self):
        tmp = self.name + ":\n"
        for key in self.sets.keys():
            tmp += str(self.sets[key])+ "\n"
        return tmp
