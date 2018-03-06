import pandas as pd
import numpy as np
import os
import pkg_resources
from pathlib import Path
from urllib import request


def get_dataframe(filename, url, sep=";", compression='infer'):
    #filename = pkg_resources.resource_filename('pyFTS', path)
    tmp_file = Path(filename)

    if tmp_file.is_file():
        return pd.read_csv(filename, sep=sep, compression=compression)
    else:
        request.urlretrieve(url, filename)
        return pd.read_csv(filename, sep=sep, compression=compression)


