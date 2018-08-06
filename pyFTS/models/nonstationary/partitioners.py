import numpy as np
from pyFTS.partitioners import partitioner
from pyFTS.models.nonstationary import common, perturbation
from pyFTS.common import FuzzySet as stationary_fs


class PolynomialNonStationaryPartitioner(partitioner.Partitioner):
    """
    Non Stationary Universe of Discourse Partitioner
    """

    def __init__(self, data, part, **kwargs):
        """"""
        super(PolynomialNonStationaryPartitioner, self).__init__(name=part.name, data=data, npart=part.partitions,
                                                                 func=part.membership_function, names=part.setnames,
                                                                 prefix=part.prefix, transformation=part.transformation,
                                                                 indexer=part.indexer, preprocess=False)

        self.sets = {}

        loc_params, wid_params = self.get_polynomial_perturbations(data, **kwargs)

        if self.ordered_sets is None and self.setnames is not None:
            self.ordered_sets = part.setnames
        else:
            self.ordered_sets = stationary_fs.set_ordered(part.sets)

        for ct, key in enumerate(self.ordered_sets):
            set = part.sets[key]
            loc_roots = np.roots(loc_params[ct])[0]
            wid_roots = np.roots(wid_params[ct])[0]
            tmp = common.FuzzySet(set.name, set.mf, set.parameters,
                           location=perturbation.polynomial,
                           location_params=loc_params[ct],
                           location_roots=loc_roots, #**kwargs)
                           width=perturbation.polynomial,
                           width_params=wid_params[ct],
                           width_roots=wid_roots, **kwargs)

            self.sets[set.name] = tmp

    def poly_width(self, par1, par2, rng, deg):
        a = np.polyval(par1, rng)
        b = np.polyval(par2, rng)
        diff = [b[k] - a[k] for k in rng]
        tmp = np.polyfit(rng, diff, deg=deg)
        return tmp

    def scale_up(self,x,pct):
        if x > 0: return x*(1+pct)
        else: return x*pct

    def scale_down(self,x,pct):
        if x > 0: return x*pct
        else: return x*(1+pct)

    def get_polynomial_perturbations(self, data, **kwargs):
        w = kwargs.get("window_size", int(len(data) / 5))
        degree = kwargs.get("degree", 2)
        xmax = [data[0]]
        tmax = [0]
        xmin = [data[0]]
        tmin = [0]

        l = len(data)

        for i in np.arange(0, l, w):
            sample = data[i:i + w]
            tx = max(sample)
            xmax.append(tx)
            tmax.append(np.ravel(np.argwhere(data == tx)).tolist()[0])
            tn = min(sample)
            xmin.append(tn)
            tmin.append(np.ravel(np.argwhere(data == tn)).tolist()[0])

        cmax = np.polyfit(tmax, xmax, deg=degree)
        cmin = np.polyfit(tmin, xmin, deg=degree)

        cmed = []

        for d in np.arange(0, degree + 1):
            cmed.append(np.linspace(cmin[d], cmax[d], self.partitions)[1:self.partitions - 1])

        loc_params = [cmin.tolist()]
        for i in np.arange(0, self.partitions - 2):
            tmp = [cmed[k][i] for k in np.arange(0, degree + 1)]
            loc_params.append(tmp)
        loc_params.append(cmax.tolist())

        rng = np.arange(0, l)

        clen = []

        for i in np.arange(1, self.partitions-1):
            tmp = self.poly_width(loc_params[i - 1], loc_params[i + 1], rng, degree)
            clen.append(tmp)

        tmp = self.poly_width(loc_params[0], loc_params[1], rng, degree)
        clen.insert(0, tmp)

        tmp = self.poly_width(loc_params[self.partitions-2], loc_params[self.partitions-1], rng, degree)
        clen.append(tmp)

        tmp = (loc_params, clen)

        return tmp

    def build(self, data):
        pass


class SimpleNonStationaryPartitioner(partitioner.Partitioner):
    """
    Non Stationary Universe of Discourse Partitioner
    """

    def __init__(self, data, part, **kwargs):
        """"""
        super(SimpleNonStationaryPartitioner, self).__init__(name=part.name, data=data, npart=part.partitions,
                                                             func=part.membership_function, names=part.setnames,
                                                             prefix=part.prefix, transformation=part.transformation,
                                                             indexer=part.indexer)#, preprocess=False)

        for key in part.sets.keys():
            set = part.sets[key]
            tmp = common.FuzzySet(set.name, set.mf, set.parameters, **kwargs)
            tmp.centroid = set.centroid

            self.sets[key] =tmp

        self.ordered_sets = stationary_fs.set_ordered(self.sets)

    def build(self, data):
        return {}


def simplenonstationary_gridpartitioner_builder(data, npart, transformation):
    from pyFTS.partitioners import Grid
    from pyFTS.models.nonstationary import perturbation, partitioners

    tmp_fs = Grid.GridPartitioner(data=data, npart=npart, transformation=transformation)
    fs = partitioners.SimpleNonStationaryPartitioner(data, tmp_fs,
                                                     location=perturbation.polynomial,
                                                     location_params=[1, 0],
                                                     location_roots=0,
                                                     width=perturbation.polynomial,
                                                     width_params=[1, 0],
                                                     width_roots=0)
    return fs
