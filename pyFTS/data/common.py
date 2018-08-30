
import pandas as pd
import numpy as np
import os
import pkg_resources
from pathlib import Path
from urllib import request


def get_dataframe(filename, url, sep=";", compression='infer'):
    """
    This method check if filename already exists, read the file and return its data.
    If the file don't already exists, it will be downloaded and decompressed.

    :param filename: dataset local filename
    :param url: dataset internet URL
    :param sep: CSV field separator
    :param compression: type of compression
    :return:  Pandas dataset
    """

    tmp_file = Path(filename)

    if tmp_file.is_file():
        return pd.read_csv(filename, sep=sep, compression=compression)
    else:
        request.urlretrieve(url, filename)
        return pd.read_csv(filename, sep=sep, compression=compression)


