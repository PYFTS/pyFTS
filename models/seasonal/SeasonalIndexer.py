import numpy as np

class SeasonalIndexer(object):
    def __init__(self,num_seasons):
        self.num_seasons = num_seasons

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


class LinearSeasonalIndexer(SeasonalIndexer):
    def __init__(self,seasons):
        super(LinearSeasonalIndexer, self).__init__(len(seasons))
        self.seasons = seasons

    def get_season_of_data(self,data):
        return self.get_season_by_index(np.arange(0,len(data)))

    def get_season_by_index(self,index):
        ret = []
        for ix in index:
            if self.num_seasons == 1:
                season = ix % self.seasons
            else:
                season = []
                for seasonality in self.seasons:
                    print("S ", seasonality)
                    tmp = ix // seasonality
                    print("T ", tmp)
                    season.append(tmp)
                #season.append(rest)

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
    def __init__(self,index_fields,index_seasons, data_fields):
        super(DataFrameSeasonalIndexer, self).__init__(len(index_seasons))
        self.fields = index_fields
        self.seasons = index_seasons
        self.data_fields = data_fields

    def get_season_of_data(self,data):
        ret = []
        for ix in data.index:
            season = []
            for c, f in enumerate(self.fields, start=0):
                if self.seasons[c] is None:
                    season.append(data[f][ix])
                else:
                    season.append(data[f][ix] // self.seasons[c])
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
        return data[self.data_fields]

    def get_index_by_season(self, indexes):
        raise Exception("Operation not available!")

    def get_data(self, data):
        return data[self.data_fields].tolist()

    def set_data(self, data, value):
        data[self.data_fields] = value
        return data