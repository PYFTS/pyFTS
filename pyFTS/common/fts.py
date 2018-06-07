import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, SortedCollection, tree, Util


class FTS(object):
    """
    Fuzzy Time Series object model
    """
    def __init__(self, **kwargs):
        """
        Create a Fuzzy Time Series model
        :param order: model order
        :param name: model name
        :param kwargs: model specific parameters
        """
        self.sets = {}
        self.flrgs = {}
        self.order = kwargs.get('order',"")
        self.shortname = kwargs.get('name',"")
        self.name = kwargs.get('name',"")
        self.detail = kwargs.get('name',"")
        self.is_high_order = False
        self.min_order = 1
        self.has_seasonality = False
        self.has_point_forecasting = True
        self.has_interval_forecasting = False
        self.has_probability_forecasting = False
        self.is_multivariate = False
        self.dump = False
        self.transformations = []
        self.transformations_param = []
        self.original_max = 0
        self.original_min = 0
        self.partitioner = kwargs.get("partitioner", None)
        if self.partitioner != None:
            self.sets = self.partitioner.sets
        self.auto_update = False
        self.benchmark_only = False
        self.indexer = None
        self.uod_clip = kwargs.get("uod_clip", True)

    def fuzzy(self, data):
        """
        Fuzzify a data point
        :param data: data point
        :return: maximum membership fuzzy set
        """
        best = {"fuzzyset": "", "membership": 0.0}

        for f in self.sets:
            fset = self.sets[f]
            if best["membership"] <= fset.membership(data):
                best["fuzzyset"] = fset.name
                best["membership"] = fset.membership(data)

        return best

    def predict(self, data, **kwargs):
        """
        Forecast using trained model
        :param data: time series with minimal length to the order of the model
        :param kwargs:

        :keyword
        type: the forecasting type, one of these values: point(default), interval or distribution.
        steps_ahead: The forecasting horizon, i. e., the number of steps ahead to forecast
        start: in the multi step forecasting, the index of the data where to start forecasting
        distributed: boolean, indicate if the forecasting procedure will be distributed in a dispy cluster
        nodes: a list with the dispy cluster nodes addresses
        :return: a numpy array with the forecasted data
        """

        if self.is_multivariate:
            ndata = data
        else:
            ndata = self.apply_transformations(data)

            if self.uod_clip:
                ndata = np.clip(ndata, self.original_min, self.original_max)

        if 'distributed' in kwargs:
            distributed = kwargs.pop('distributed')
        else:
            distributed = False

        if distributed is None or distributed == False:

            if 'type' in kwargs:
                type = kwargs.pop("type")
            else:
                type = 'point'

            steps_ahead = kwargs.get("steps_ahead", None)

            if steps_ahead == None or steps_ahead == 1:
                if type == 'point':
                    ret = self.forecast(ndata, **kwargs)
                elif type == 'interval':
                    ret = self.forecast_interval(ndata, **kwargs)
                elif type == 'distribution':
                    ret = self.forecast_distribution(ndata, **kwargs)
            elif steps_ahead > 1:
                if type == 'point':
                    ret = self.forecast_ahead(ndata, steps_ahead, **kwargs)
                elif type == 'interval':
                    ret = self.forecast_ahead_interval(ndata, steps_ahead, **kwargs)
                elif type == 'distribution':
                    ret = self.forecast_ahead_distribution(ndata, steps_ahead, **kwargs)

            if not ['point', 'interval', 'distribution'].__contains__(type):
                raise ValueError('The argument \'type\' has an unknown value.')

        else:

            nodes = kwargs.get("nodes", ['127.0.0.1'])
            num_batches = kwargs.get('num_batches', 10)

            ret = Util.distributed_predict(self, kwargs, nodes, ndata, num_batches)

        if not self.is_multivariate:
            kwargs['type'] = type
            ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]], **kwargs)

        return ret

    def forecast(self, data, **kwargs):
        """
        Point forecast one step ahead 
        :param data: time series with minimal length to the order of the model
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform one step ahead point forecasts!')

    def forecast_interval(self, data, **kwargs):
        """
        Interval forecast one step ahead
        :param data: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform one step ahead interval forecasts!')

    def forecast_distribution(self, data, **kwargs):
        """
        Probabilistic forecast one step ahead
        :param data: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform one step ahead distribution forecasts!')

    def forecast_ahead(self, data, steps, **kwargs):
        """
        Point forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        ret = []
        for k in np.arange(0,steps):
            tmp = self.forecast(data[-self.order:], **kwargs)

            if isinstance(tmp,(list, np.ndarray)):
                tmp = tmp[0]

            ret.append(tmp)
            data.append_rhs(tmp)

        return ret

    def forecast_ahead_interval(self, data, steps, **kwargs):
        """
        Interval forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform multi step ahead interval forecasts!')

    def forecast_ahead_distribution(self, data, steps, **kwargs):
        """
        Probabilistic forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform multi step ahead distribution forecasts!')

    def train(self, data, **kwargs):
        """
        
        :param data: 
        :param sets: 
        :param order: 
        :param parameters: 
        :return: 
        """
        pass

    def fit(self, ndata, **kwargs):
        """

        :param data: the training time series
        :param kwargs:

        :keyword
        num_batches: split the training data in num_batches to save memory during the training process
        save_model: save final model on disk
        batch_save: save the model between each batch
        file_path: path to save the model
        distributed: boolean, indicate if the training procedure will be distributed in a dispy cluster
        nodes: a list with the dispy cluster nodes addresses

        :return:
        """

        import datetime

        if self.is_multivariate:
            data = ndata
        else:
            data = self.apply_transformations(ndata)

            self.original_min = np.nanmin(data)
            self.original_max = np.nanmax(data)

        if 'sets' in kwargs:
            self.sets = kwargs.pop('sets')

        if 'partitioner' in kwargs:
            self.partitioner = kwargs.pop('partitioner')

        if (self.sets is None or len(self.sets) == 0) and not self.benchmark_only and not self.is_multivariate:
            if self.partitioner is not None:
                self.sets = self.partitioner.sets
            else:
                raise Exception("Fuzzy sets were not provided for the model. Use 'sets' parameter or 'partitioner'. ")

        if 'order' in kwargs:
            self.order = kwargs.pop('order')

        dump = kwargs.get('dump', None)

        num_batches = kwargs.get('num_batches', None)

        save = kwargs.get('save_model', False)  # save model on disk

        batch_save = kwargs.get('batch_save', False) #save model between batches

        file_path = kwargs.get('file_path', None)

        distributed = kwargs.get('distributed', False)

        batch_save_interval = kwargs.get('batch_save_interval', 10)

        if distributed:
            nodes = kwargs.get('nodes', False)
            train_method = kwargs.get('train_method', Util.simple_model_train)
            Util.distributed_train(self, train_method, nodes, type(self), data, num_batches, {},
                                   batch_save=batch_save, file_path=file_path,
                                   batch_save_interval=batch_save_interval)
        else:

            if dump == 'time':
                print("[{0: %H:%M:%S}] Start training".format(datetime.datetime.now()))

            if num_batches is not None:
                n = len(data)
                batch_size = int(n / num_batches)
                bcount = 1

                rng = range(self.order, n, batch_size)

                if dump == 'tqdm':
                    from tqdm import tqdm

                    rng = tqdm(rng)

                for ct in rng:
                    if dump == 'time':
                        print("[{0: %H:%M:%S}] Starting batch ".format(datetime.datetime.now()) + str(bcount))
                    if self.is_multivariate:
                        mdata = data.iloc[ct - self.order:ct + batch_size]
                    else:
                        mdata = data[ct - self.order : ct + batch_size]

                    self.train(mdata, **kwargs)

                    if batch_save:
                        Util.persist_obj(self,file_path)

                    if dump == 'time':
                        print("[{0: %H:%M:%S}] Finish batch ".format(datetime.datetime.now()) + str(bcount))

                    bcount += 1

            else:
                self.train(data, **kwargs)

            if dump == 'time':
                print("[{0: %H:%M:%S}] Finish training".format(datetime.datetime.now()))

        if save:
            Util.persist_obj(self, file_path)

    def clone_parameters(self, model):
        self.order = model.order
        self.shortname = model.shortname
        self.name = model.name
        self.detail = model.detail
        self.is_high_order = model.is_high_order
        self.min_order = model.min_order
        self.has_seasonality = model.has_seasonality
        self.has_point_forecasting = model.has_point_forecasting
        self.has_interval_forecasting = model.has_interval_forecasting
        self.has_probability_forecasting = model.has_probability_forecasting
        self.is_multivariate = model.is_multivariate
        self.dump = model.dump
        self.transformations = model.transformations
        self.transformations_param = model.transformations_param
        self.original_max = model.original_max
        self.original_min = model.original_min
        self.partitioner = model.partitioner
        self.sets = model.sets
        self.auto_update = model.auto_update
        self.benchmark_only = model.benchmark_only
        self.indexer = model.indexer

    def merge(self, model):
        for key in model.flrgs.keys():
            flrg = model.flrgs[key]
            if flrg.get_key() not in self.flrgs:
                self.flrgs[flrg.get_key()] = flrg
            else:
                if isinstance(flrg.RHS, (list, set)):
                    for k in flrg.RHS:
                        self.flrgs[flrg.get_key()].append_rhs(k)
                elif isinstance(flrg.RHS, dict):
                    for k in flrg.RHS.keys():
                        self.flrgs[flrg.get_key()].append_rhs(flrg.RHS[k])
                else:
                    self.flrgs[flrg.get_key()].append_rhs(flrg.RHS)

    def append_transformation(self, transformation):
        if transformation is not None:
            self.transformations.append(transformation)

    def apply_transformations(self, data, params=None, updateUoD=False, **kwargs):
        ndata = data
        if updateUoD:
            if min(data) < 0:
                self.original_min = min(data) * 1.1
            else:
                self.original_min = min(data) * 0.9

            if max(data) > 0:
                self.original_max = max(data) * 1.1
            else:
                self.original_max = max(data) * 0.9

        if len(self.transformations) > 0:
            if params is None:
                params = [ None for k in self.transformations]

            for c, t in enumerate(self.transformations, start=0):
                ndata = t.apply(ndata,params[c])

        return ndata

    def apply_inverse_transformations(self, data, params=None, **kwargs):
        if len(self.transformations) > 0:
            if params is None:
                params = [None for k in self.transformations]

            for c, t in enumerate(reversed(self.transformations), start=0):
                ndata = t.inverse(data, params[c], **kwargs)

            return ndata
        else:
            return data

    def get_UoD(self):
        return [self.original_min, self.original_max]

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            tmp = tmp + str(self.flrgs[r]) + "\n"
        return tmp

    def __len__(self):
       return len(self.flrgs)

    def len_total(self):
        return sum([len(k) for k in self.flrgs])







