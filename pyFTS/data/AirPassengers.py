import pandas as pd
import numpy as np


def get_data():
    passengers = pd.read_csv("DataSets/AirPassengers.csv", sep=",")
    passengers = np.array(passengers["Passengers"])
    return passengers
