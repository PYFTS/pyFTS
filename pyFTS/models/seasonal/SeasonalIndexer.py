import numpy as np
import pandas as pd
from pyFTS.models.seasonal import common


class SeasonalIndexer(object):
    """
    Seasonal Indexer. Responsible to find the seasonal index of a data point inside its data set
    """
    def __init__(self,num_seasons, **kwargs):
        self.num_seasons = num_seasons
        self.name = kwargs.get("name","")

    def get_season_of_data(self,data):
        pass

    def get_season_by_index(self,inde):
        pass

    def get_data_by_season(self, data, indexes):
        pass

    def get_index_by_season(self, indexes):
        pass

    def get_data(self, data):
        pass

    def get_index(self, data):
        pass


class LinearSeasonalIndexer(SeasonalIndexer):
    """Use the data array/list position to index the seasonality """
    def __init__(self, seasons, units, ignore=None, **kwargs):
        """
        Indexer for array/list position
        :param seasons: A list with the season group (i.e: 7 for week, 30 for month, etc)
        :param units: A list with the units used for each season group, the default is 1 for each
        :param ignore:
        :param kwargs:
        """
        super(LinearSeasonalIndexer, self).__init__(len(seasons), **kwargs)
        self.seasons = seasons
        self.units = units
        self.ignore = ignore

    def get_season_of_data(self,data):
        return self.get_season_by_index(np.arange(0, len(data)).tolist())

    def get_season_by_index(self, index):
        ret = []
        if not isinstance(index, (list, np.ndarray)):
            if self.num_seasons == 1:
                season = (index // self.units[0]) % self.seasons[0]
            else:
                season = []
                for ct, seasonality in enumerate(self.seasons, start=0):
                    tmp = (index // self.units[ct]) % self.seasons[ct]
                    if not self.ignore[ct]:
                        season.append(tmp)
            ret.append(season)
        else:
            for ix in index:
                if self.num_seasons == 1:
                    season = (ix  // self.units[0]) % self.seasons[0]
                else:
                    season = []
                    for ct, seasonality in enumerate(self.seasons, start=0):
                        tmp = (ix  // self.units[ct]) % self.seasons[ct]
                        if not self.ignore[ct]:
                            season.append(tmp)
                ret.append(season)

        return ret

    def get_index_by_season(self, indexes):
        ix = 0;

        for count,season in enumerate(self.seasons):
            ix += season*(indexes[count])

        #ix += indexes[-1]

        return ix

    def get_data(self, data):
        return data


class DataFrameSeasonalIndexer(SeasonalIndexer):
    """Use the Pandas.DataFrame index position to index the seasonality """
    def __init__(self,index_fields,index_seasons, data_field,**kwargs):
        """

        :param index_fields: DataFrame field to use as index
        :param index_seasons: A list with the season group, i. e., multiples of positions that are considered a season (i.e: 7 for week, 30 for month, etc)
        :param data_fields: DataFrame field to use as data
        :param kwargs:
        """
        super(DataFrameSeasonalIndexer, self).__init__(len(index_seasons), **kwargs)
        self.fields = index_fields
        self.seasons = index_seasons
        self.data_field = data_field

    def get_season_of_data(self,data):
        #data = data.copy()
        ret = []
        for ix in data.index:
            season = []
            for c, f in enumerate(self.fields, start=0):
                if self.seasons[c] is None:
                    season.append(data[f][ix])
                else:
                    a = data[f][ix]
                    season.append(a // self.seasons[c])
            ret.append(season)
        return ret

    def get_season_by_index(self,index):
        raise Exception("Operation not available!")

    def get_data_by_season(self, data, indexes):
        for season in indexes:
            for c, f in enumerate(self.fields, start=0):
                if self.seasons[c] is None:
                    data = data[data[f]== season[c]]
                else:
                    data = data[(data[f] // self.seasons[c]) == season[c]]
        return data[self.data_field]

    def get_index_by_season(self, indexes):
        raise Exception("Operation not available!")

    def get_data(self, data):
        return data[self.data_field].tolist()

    def set_data(self, data, value):
        data.loc[:,self.data_field] = value
        return data


class DateTimeSeasonalIndexer(SeasonalIndexer):
    """Use a Pandas.DataFrame date field to index the seasonality """
    def __init__(self,date_field, index_fields, index_seasons, data_field,**kwargs):
        """

        :param date_field: DataFrame field that contains the datetime field used on index
        :param index_fields: List with commom.DataTime fields
        :param index_seasons: Multiples of index_fields, the default is 1
        :param data_field: DataFrame field with the time series data
        :param kwargs:
        """
        super(DateTimeSeasonalIndexer, self).__init__(len(index_seasons), **kwargs)
        self.fields = index_fields
        self.seasons = index_seasons
        self.data_field = data_field
        self.date_field = date_field

    def get_season_of_data(self, data):

        ret = []

        if isinstance(data, pd.DataFrame):
            for ix in data.index:
                date = data[self.date_field][ix]
                season = []
                for c, f in enumerate(self.fields, start=0):
                    tmp = common.strip_datepart(date, f)
                    if self.seasons[c] is not None:
                        tmp = tmp // self.seasons[c]
                    season.append(tmp)
                ret.append(season)

        elif isinstance(data, pd.Series):
            date = data[self.date_field]
            season = []
            for c, f in enumerate(self.fields, start=0):
                season.append(common.strip_datepart(date, f, self.seasons[c]))
            ret.append(season)

        return ret

    def get_season_by_index(self, index):
        raise Exception("Operation not available!")

    def get_data_by_season(self, data, indexes):
        raise Exception("Operation not available!")

    def get_index_by_season(self, indexes):
        raise Exception("Operation not available!")

    def get_data(self, data):
        return data[self.data_field].tolist()

    def get_index(self, data):
        return data[self.date_field].tolist() if isinstance(data, pd.DataFrame) else data[self.date_field]

    def set_data(self, data, value):
        raise Exception("Operation not available!")


