import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


def distance(x, y):
    if isinstance(x, list):
        tmp = functools.reduce(operator.add, [(x[k] - y[k]) ** 2 for k in range(0, len(x))])
    else:
        tmp = (x - y) ** 2
    return math.sqrt(tmp)


def c_means(k, dados, tam):
    # Inicializa as centróides escolhendo elementos aleatórios dos conjuntos
    centroides = [dados[rnd.randint(0, len(dados)-1)] for kk in range(0, k)]

    grupos = [-1 for x in range(0, len(dados))]

    it_semmodificacao = 0

    # para cada instância
    iteracoes = 0
    while iteracoes < 1000 and it_semmodificacao < 10:
        inst_count = 0

        modificacao = False

        for instancia in dados:

            # verifica a distância para cada centroide
            grupo_count = 0
            dist = 10000

            grupotmp = grupos[inst_count]

            for grupo in centroides:
                tmp = distance(instancia, grupo)
                if tmp < dist:
                    dist = tmp
                    # associa a a centroide de menor distância à instância
                    grupos[inst_count] = grupo_count
                grupo_count = grupo_count + 1

            if grupotmp != grupos[inst_count]:
                modificacao = True

            inst_count = inst_count + 1

        if not modificacao:
            it_semmodificacao = it_semmodificacao + 1
        else:
            it_semmodificacao = 0

        # atualiza cada centroide com base nos valores médios de todas as instâncias à ela associadas
        grupo_count = 0
        for grupo in centroides:
            total_inst = functools.reduce(operator.add, [1 for xx in grupos if xx == grupo_count], 0)
            if total_inst > 0:
                if tam > 1:
                    for count in range(0, tam):
                        soma = functools.reduce(operator.add,
                                                [dados[kk][count] for kk in range(0, len(dados)) if
                                                 grupos[kk] == grupo_count])
                        centroides[grupo_count][count] = soma / total_inst
                else:
                    soma = functools.reduce(operator.add,
                                            [dados[kk] for kk in range(0, len(dados)) if grupos[kk] == grupo_count])
                    centroides[grupo_count] = soma / total_inst
            grupo_count = grupo_count + 1

        iteracoes = iteracoes + 1

    return centroides


class CMeansPartitioner(partitioner.Partitioner):
    def __init__(self, **kwargs):
        super(CMeansPartitioner, self).__init__(name="CMeans", **kwargs)

    def build(self, data):
        sets = {}
        centroides = c_means(self.partitions, data, 1)
        centroides.append(self.max)
        centroides.append(self.min)
        centroides = list(set(centroides))
        centroides.sort()
        for c in np.arange(1, len(centroides) - 1):
            _name = self.get_name(c)
            sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                 [round(centroides[c - 1], 3), round(centroides[c], 3), round(centroides[c + 1], 3)],
                                 round(centroides[c], 3))

        return sets
