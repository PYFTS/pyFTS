#--------------------
#BDMEP - INMET
#--------------------
#Estação           : BELO HORIZONTE - MG (OMM: 83587)
#Latitude  (graus) : -19.93
#Longitude (graus) : -43.93
#Altitude  (metros): 915.00
#Estação Operante
#Inicio de operação: 03/03/1910
#Periodo solicitado dos dados: 01/01/2000 a 31/12/2012
#Os dados listados abaixo são os que encontram-se digitados no BDMEP
#Hora em UTC

# http://www.inmet.gov.br

import pandas as pd
import numpy as np
import os
import pkg_resources


def get_dataframe():
    filename = pkg_resources.resource_filename('pyFTS', 'data/INMET.csv.bz2')
    dat = pd.read_csv(filename, sep=";", compression='bz2')
    dat["DataHora"] = pd.to_datetime(dat["DataHora"], format='%d/%m/%Y %H:%M')
    return dat
