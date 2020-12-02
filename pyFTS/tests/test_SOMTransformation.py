import unittest
from pyFTS.common.transformations.som import SOMTransformation
import pandas as pd
import os
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_apply(self):
        data = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        som = self.som_transformer_trained()
        transformed = som.apply(data=pd.DataFrame(data))
        uniques = np.unique(transformed)

        self.assertEqual(1, len(uniques.shape))
        self.assertEqual(3, transformed.values.shape[1])

    def test_save_net(self):
        som_transformer = self.som_transformer_trained()

        filename = 'test_net.npy'
        som_transformer.save_net(filename)
        files = os.listdir()

        if filename in files:
            is_in_files = True
            os.remove(filename)
        else:
            is_in_files = False

        self.assertEqual(True, is_in_files)

    # def

    def test_train(self):
        self.assertEqual()

    @staticmethod
    def simple_dataset():
        data = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
        df = pd.DataFrame(data)
        return df

    def som_transformer_trained(self):
        data = self.simple_dataset()
        som_transformer = SOMTransformation(grid_dimension=(2, 2))
        som_transformer.train(data=data, epochs=100)
        return som_transformer

if __name__ == '__main__':
    unittest.main()
