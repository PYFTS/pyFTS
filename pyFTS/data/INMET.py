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

from pyFTS.data import common
import pandas as pd


def get_dataframe():
    dat = common.get_dataframe('INMET.csv.bz2',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/INMET.csv.bz2',
                               sep=";", compression='bz2')
    dat["DataHora"] = pd.to_datetime(dat["DataHora"], format='%d/%m/%Y %H:%M')
    return dat
