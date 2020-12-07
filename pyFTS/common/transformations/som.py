"""
Kohonen Self Organizing Maps for Fuzzy Time Series
"""
import pandas as pd
import SimpSOM as sps
from pyFTS.models.multivariate import wmvfts
from typing import Tuple
from pyFTS.common.Transformations import Transformation
from typing import List

class SOMTransformation(Transformation):
    def __init__(self,
                 grid_dimension: Tuple,
                 **kwargs):
        # SOM attributes
        self.load_file = kwargs.get('loadFile')
        self.net: sps.somNet = None
        self.data: pd.DataFrame = None
        self.grid_dimension: Tuple = grid_dimension
        self.pbc = kwargs.get('PBC', True)

        # debug attributes
        self.name = 'Kohonen Self Organizing Maps FTS'
        self.shortname = 'SOM-FTS'

    def apply(self,
              data: pd.DataFrame,
              endogen_variable=None,
              names: Tuple[str] = ('x', 'y'),
              param=None,
              **kwargs):
        """
        Transform a M-dimensional dataset into a 3-dimensional dataset, where one dimension is the endogen variable
        If endogen_variable = None, the last column will be the endogen_variable.
        Args:
            data (pd.DataFrame): M-Dimensional dataset
            endogen_variable (str):  column of dataset
            names (Tuple): names for new columns created by SOM Transformation.
            param:
            **kwargs: params of SOM's train process
                percentage_train (float). Percentage of dataset that will be used for train SOM network. default: 0.7
                leaning_rate (float): leaning rate of SOM network. default: 0.01
                epochs: epochs of SOM network. default: 10000
        Returns:

        """

        if endogen_variable not in data.columns:
            endogen_variable = None
        cols = data.columns[:-1] if endogen_variable is None else [col for col in data.columns if
                                                                   col != endogen_variable]
        if self.net is None:
            train = data[cols]
            self.train(data=train, **kwargs)
        new_data = self.net.project(data[cols].values)
        new_data = pd.DataFrame(new_data, columns=names)
        endogen = endogen_variable if endogen_variable is not None else data.columns[-1]
        new_data[endogen] = data[endogen].values
        return new_data

    def __repr__(self):
        status = "is trained" if self.net is not None else "not trained"
        return f'{self.name}-{status}'

    def __str__(self):
        return self.name

    def __del__(self):
        del self.net

    def train(self,
              data: pd.DataFrame,
              percentage_train: float = .7,
              leaning_rate: float = 0.01,
              epochs: int = 10000):
        data = data.dropna()
        self.data = data.values
        limit = round(len(self.data) * percentage_train)
        train = self.data[:limit]
        x, y = self.grid_dimension
        self.net = sps.somNet(x, y, train, PBC=self.pbc, loadFile=self.load_file)
        self.net.train(startLearnRate=leaning_rate,
                       epochs=epochs)


    def save_net(self,
                 filename: str = "SomNet trained"):
        self.net.save(filename)
        self.load_file = filename

    def show_grid(self,
                  graph_type: str = 'nodes_graph',
                  **kwargs):
        if graph_type == 'nodes_graph':
            colnum = kwargs.get('colnum', 0)
            self.net.nodes_graph(colnum=colnum)
        else:
            self.net.diff_graph()
