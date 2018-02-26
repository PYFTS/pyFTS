import pandas as pd
import numpy as np
import os
import pkg_resources

def get_data():
    #data_path = os.path.dirname(__file__)
    #filename = os.path.join(data_path,"Enrollments.csv")
    filename = pkg_resources.resource_filename('pyFTS', 'data/Enrollments.csv')
    enrollments = pd.read_csv(filename, sep=";")
    enrollments = np.array(enrollments["Enrollments"])
    return enrollments
