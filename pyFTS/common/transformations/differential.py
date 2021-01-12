from pyFTS.common.transformations.transformation import Transformation 


class Differential(Transformation):
    """
    Differentiation data transform

    y'(t) = y(t) - y(t-1)
    y(t) =  y(t-1)  + y'(t)
    """
    def __init__(self, lag):
        super(Differential, self).__init__()
        self.lag = lag
        self.minimal_length = 2
        self.name = 'Diff'

    @property
    def parameters(self):
        return self.lag

    def apply(self, data, param=None, **kwargs):
        if param is not None:
            self.lag = param

        if not isinstance(data, (list, np.ndarray, np.generic)):
            data = [data]

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        n = len(data)
        diff = [data[t] - data[t - self.lag] for t in np.arange(self.lag, n)]
        for t in np.arange(0, self.lag): diff.insert(0, 0)
        return diff

    def inverse(self, data, param, **kwargs):

        type = kwargs.get("type","point")
        steps_ahead = kwargs.get("steps_ahead", 1)

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        n = len(data)

#        print(n)
#        print(len(param))

        if steps_ahead == 1:
            if type == "point":
                inc = [data[t] + param[t] for t in np.arange(0, n)]
            elif type == "interval":
                inc = [[data[t][0] + param[t], data[t][1] + param[t]] for t in np.arange(0, n)]
            elif type == "distribution":
                for t in np.arange(0, n):
                    data[t].differential_offset(param[t])
                inc = data
        else:
            if type == "point":
                inc = [data[0] + param[0]]
                for t in np.arange(1, steps_ahead):
                    inc.append(data[t] + inc[t-1])
            elif type == "interval":
                inc = [[data[0][0] + param[0], data[0][1] + param[0]]]
                for t in np.arange(1, steps_ahead):
                    inc.append([data[t][0] + np.nanmean(inc[t-1]), data[t][1] + np.nanmean(inc[t-1])])
            elif type == "distribution":
                data[0].differential_offset(param[0])
                for t in np.arange(1, steps_ahead):
                    ex = data[t-1].expected_value()
                    data[t].differential_offset(ex)
                inc = data

        if n == 1:
            return inc[0]
        else:
            return inc