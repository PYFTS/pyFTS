import pandas as pd
import numpy as np
import os
import pkg_resources


def get_data():
    filename = pkg_resources.resource_filename('pyFTS', 'data/Enrollments.csv')
    enrollments = pd.read_csv(filename, sep=";")
    enrollments = np.array(enrollments["Enrollments"])
    return enrollments
