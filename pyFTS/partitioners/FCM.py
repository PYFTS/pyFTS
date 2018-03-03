import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


# import CMeans

# S. T. Li, Y. C. Cheng, and S. Y. Lin, “A FCM-based deterministic forecasting model for fuzzy time series,”
# Comput. Math. Appl., vol. 56, no. 12, pp. 3052–3063, Dec. 2008. DOI: 10.1016/j.camwa.2008.07.033.

def fuzzy_distance(x, y):
    if isinstance(x, list):
        tmp = functools.reduce(operator.add, [(x[k] - y[k]) ** 2 for k in range(0, len(x))])
    else:
        tmp = (x - y) ** 2
    return math.sqrt(tmp)


def membership(val, vals):
    soma = 0
    for k in vals:
        if k == 0:
            k = 1
        soma = soma + (val / k) ** 2

    return soma


def fuzzy_cmeans(k, dados, tam, m, deltadist=0.001):
    tam_dados = len(dados)

    # Inicializa as centróides escolhendo elementos aleatórios dos conjuntos
    centroides = [dados[rnd.randint(0, tam_dados - 1)] for kk in range(0, k)]

    # Tabela de pertinência das instâncias aos grupos
    grupos = [[0 for kk in range(0, k)] for xx in range(0, tam_dados)]

    alteracaomedia = 1000

    m_exp = 1 / (m - 1)

    # para cada instância
    iteracoes = 0

    while iteracoes < 1000 and alteracaomedia > deltadist:

        alteracaomedia = 0

        # verifica a distância para cada centroide
        # Atualiza a pertinencia daquela instância para cada um dos grupos

        inst_count = 0
        for instancia in dados:

            dist_grupos = [0 for xx in range(0, k)]

            grupo_count = 0
            for grupo in centroides:
                dist_grupos[grupo_count] = fuzzy_distance(grupo, instancia)
                grupo_count = grupo_count + 1

            dist_grupos_total = functools.reduce(operator.add, [xk for xk in dist_grupos])

            for grp in range(0, k):
                if dist_grupos[grp] == 0:
                    grupos[inst_count][grp] = 1
                else:
                    grupos[inst_count][grp] = 1 / membership(dist_grupos[grp], dist_grupos)
                    # grupos[inst_count][grp] = 1/(dist_grupos[grp] / dist_grupos_total)
                    # grupos[inst_count][grp] = (1/(dist_grupos[grp]**2))**m_exp / (1/(dist_grupos_total**2))**m_exp

            inst_count = inst_count + 1

        # return centroides

        # atualiza cada centroide com base na Média de todos os padrões ponderados pelo grau de pertinência

        grupo_count = 0
        for grupo in centroides:
            if tam > 1:
                oldgrp = [xx for xx in grupo]
                for atr in range(0, tam):
                    soma = functools.reduce(operator.add,
                                            [grupos[xk][grupo_count] * dados[xk][atr] for xk in range(0, tam_dados)])
                    norm = functools.reduce(operator.add, [grupos[xk][grupo_count] for xk in range(0, tam_dados)])
                    centroides[grupo_count][atr] = soma / norm
            else:
                oldgrp = grupo
                soma = functools.reduce(operator.add,
                                        [grupos[xk][grupo_count] * dados[xk] for xk in range(0, tam_dados)])
                norm = functools.reduce(operator.add, [grupos[xk][grupo_count] for xk in range(0, tam_dados)])
                centroides[grupo_count] = soma / norm

            alteracaomedia = alteracaomedia + fuzzy_distance(oldgrp, grupo)
            grupo_count = grupo_count + 1

        alteracaomedia = alteracaomedia / k
        iteracoes = iteracoes + 1

    return centroides


class FCMPartitioner(partitioner.Partitioner):
    """
    
    """

    def __init__(self, **kwargs):
        super(FCMPartitioner, self).__init__(name="FCM", **kwargs)

    def build(self, data):
        sets = {}

        centroids = fuzzy_cmeans(self.partitions, data, 1, 2)
        centroids.append(self.max)
        centroids.append(self.min)
        centroids = list(set(centroids))
        centroids.sort()
        for c in np.arange(1, len(centroids) - 1):
            _name = self.get_name(c)
            if self.membership_function == Membership.trimf:
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                                [round(centroids[c - 1], 3), round(centroids[c], 3),
                                                 round(centroids[c + 1], 3)],
                                                round(centroids[c], 3))
            elif self.membership_function == Membership.trapmf:
                q1 = (round(centroids[c], 3) - round(centroids[c - 1], 3)) / 2
                q2 = (round(centroids[c + 1], 3) - round(centroids[c], 3)) / 2
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                                [round(centroids[c - 1], 3), round(centroids[c], 3) - q1,
                                                 round(centroids[c], 3) + q2, round(centroids[c + 1], 3)],
                                                round(centroids[c], 3))

        return sets
