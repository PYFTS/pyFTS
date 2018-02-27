import pandas as pd
import numpy as np
import pkg_resources


def get_data():
    filename = pkg_resources.resource_filename('pyFTS', 'data/AirPassengers.csv')
    passengers = pd.read_csv(filename, sep=",")
    passengers = np.array(passengers["Passengers"])
    return passengers
