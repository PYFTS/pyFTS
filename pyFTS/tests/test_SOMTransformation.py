import unittest
from pyFTS.common.transformations.som import SOMTransformation
import pandas as pd
import os
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_apply_without_column_names(self):
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

    def test_apply_with_column_names(self):
        df = self.simple_dataset()
        df.columns = ['a', 'b', 'c', 'd', 'e']
        som = SOMTransformation(grid_dimension=(2, 2))
        result = som.apply(df, endogen_variable='a')
        result.dropna(inplace=True)
        self.assertEqual(5, len(result))
        self.assertEqual(3, len(result.columns))


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

    def test_train_with_invalid_values_should_remove_nan_row(self):
        data = [
            [1,             1, float('nan'), 1, 1],
            [1,             1, 1,            1, 0],
            [1,             1, 1,            0, 0],
            [float('nan'),  1, 0,            0, 0],
            [1,             0, 0,            0, 0],
        ]
        df = pd.DataFrame(data)
        som = SOMTransformation(grid_dimension=(2, 2))
        som.train(data=df)

        self.assertEqual(3, len(som.data))
        self.assertEqual(5, len(df.columns))

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
