"""
Kohonen Self Organizing Maps for Fuzzy Time Series
"""
import pandas as pd
import SimpSOM as sps
from pyFTS.models.multivariate import wmvfts
from typing import Tuple
from pyFTS.common.Transformations import Transformation


class SOMTransformation(Transformation):
    def __init__(self,
                 grid_dimension: Tuple,
                 **kwargs):
        # SOM attributes
        self.net: sps.somNet = None
        self.data: pd.DataFrame = None
        self.grid_dimension: Tuple = grid_dimension
        self.pbc = kwargs.get('PBC', True)

        # debug attributes
        self.name = 'Kohonen Self Organizing Maps FTS'
        self.shortname = 'SOM-FTS'

    # def apply(self, data, endogen_variable, param, **kwargs): #TODO(CASCALHO) MELHORAR DOCSTRING
    #     """
    #     Transform dataset from M-DIMENSION to 3-dimension
    #     """
    #     pass

    def __repr__(self):
        status = "is trained" if self.is_trained else "not trained"
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
        self.data = data.values
        limit = round(len(self.data) * percentage_train)
        train = self.data[:limit]
        x, y = self.grid_dimension
        self.net = sps.somNet(x, y, train, PBC=self.pbc)
        self.net.train(startLearnRate=leaning_rate,
                       epochs=epochs)

    def save_net(self,
                 filename: str = "SomNet trained"):
        self.net.save(filename)

    def show_grid(self,
                  graph_type: str = 'nodes_graph',
                  **kwargs):
        if graph_type == 'nodes_graph':
            colnum = kwargs.get('colnum', 0)
            self.net.nodes_graph(colnum=colnum)
        else:
            self.net.diff_graph()


"""
Requisitos
    - apply(herdado de transformations): transforma os conjunto de dados 
    - inverse - não é necessária 
"""
