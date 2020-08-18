# pyFTS - Fuzzy Time Series for Python

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/pyFTS/)

## What is pyFTS Library?

This package is intended for students, researchers, data scientists or whose want to exploit the Fuzzy Time Series methods. These methods provide simple, easy to use, computationally cheap and human-readable models, suitable for statistic laymans to experts.

This project is continously under improvement and contributors are well come.

## How to reference pyFTS?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.597359.svg)](https://doi.org/10.5281/zenodo.597359)

 Silva, P. C. L. et al. *pyFTS: Fuzzy Time Series for Python.* Belo Horizonte. 2018. DOI: 10.5281/zenodo.597359. Url: <http://doi.org/10.5281/zenodo.597359>

## How to install pyFTS?

First of all pyFTS was developed and tested with Python 3.6. To install pyFTS using pip tool

```
pip install -U pyFTS
```

Ou pull directly from the GitHub repo:

```
pip install -U git+https://github.com/PYFTS/pyFTS
```

## What are Fuzzy Time Series (FTS)?
Fuzzy Time Series (FTS) are non parametric methods for time series forecasting based on Fuzzy Theory.  The original method was proposed by [1] and improved later by many researchers. The general approach of the FTS methods, based on [2] is listed below:

1. **Data preprocessing**: Data transformation functions contained at [pyFTS.common.Transformations](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/common/Transformations.py), like differentiation, Box-Cox, scaling and normalization.

2. **Universe of Discourse Partitioning**: This is the most important step. Here, the range of values of the numerical time series *Y(t)* will be splited in overlapped intervals and for each interval will be created a Fuzzy Set. This step is performed by pyFTS.partition module and its classes (for instance GridPartitioner, EntropyPartitioner, etc). The main parameters are:
 - the number of intervals
 - which fuzzy membership function (on [pyFTS.common.Membership](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/common/Membership.py))
 - partition scheme ([GridPartitioner](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/Grid.py), [EntropyPartitioner](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/Entropy.py)[3], [FCMPartitioner](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/FCM.py), [CMeansPartitioner](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/CMeans.py), [HuarngPartitioner](https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/Huarng.py)[4])
 
 Check out the jupyter notebook on [notebooks/Partitioners.ipynb](https://github.com/PYFTS/notebooks/blob/master/Partitioners.ipynb) for sample codes.
 
3. **Data Fuzzyfication**: Each data point of the numerical time series *Y(t)* will be translated to a fuzzy representation (usually one or more fuzzy sets), and then a fuzzy time series *F(t)* is created.

4. **Generation of Fuzzy Rules**: In this step the temporal transition rules are created. These rules depends on the method and their characteristics:
- *order*: the number of time lags used on forecasting
- *weights*: the weighted models introduce weights on fuzzy rules for smoothing [5],[6],[7]
- *seasonality*: seasonality models depends [8]
- *steps ahead*: the number of steps ahed to predict. Almost all standard methods are based on one-step-ahead forecasting
- *forecasting type*: Almost all standard methods are point-based, but pyFTS also provides intervalar and probabilistic forecasting methods.

5. **Forecasting**: The forecasting step takes a sample (with minimum length equal to the model's order) and generate a fuzzy outputs (fuzzy set(s)) for the next time ahead. 

6. **Defuzzyfication**: This step transform the fuzzy forecast into a real number.

7. **Data postprocessing**: The inverse operations of step 1.

## Usage examples

There is nothing better than good code examples to start. [Then check out the demo Jupyter Notebooks of the implemented method os pyFTS!](https://github.com/PYFTS/notebooks).

A Google Colab example can also be found [here](https://drive.google.com/file/d/1zRBCHXOawwgmzjEoKBgmvBqkIrKxuaz9/view?usp=sharing).

## MINDS - Machine Intelligence And Data Science Lab

This tool is result of collective effort of [MINDS Lab](http://www.minds.eng.ufmg.br/), headed by Prof. Frederico Gadelha Guimaraes. Some of research on FTS which was developed under pyFTS:

- 2020
  - ORANG, Omid; Solar Energy Forecasting With Fuzzy Time Series Using High-Order Fuzzy Cognitive Maps. IEEE World Congress On Computational Intelligence 2020 (WCCI).
  - ALYOUSIFI, Y; FAYE, Othman M; SOKKALINGAM, I; SILVA, P. Markov Weighted Fuzzy Time-Series Model Based on an Optimum Partition Method for Forecasting Air Pollution. International Journal of Fuzzy Systems, 2020. http://doi.org/10.1007/s40815-020-00841-w     
  - SILVA, Petrônio CL et al. Forecasting in Non-stationary Environments with Fuzzy Time Series. https://arxiv.org/abs/2004.12554
  - SILVA, Petrônio CL et al. Distributed Evolutionary Hyperparameter Optimization for Fuzzy Time Series. IEEE Transactions on Network and Service Management, 2020. http://doi.org/10.1109/TNSM.2020.2980289
  - ALYOUSIFI, Yousif et al. Predicting Daily Air Pollution Index Based on Fuzzy Time Series Markov Chain Model. Symmetry, v. 12, n. 2, p. 293, 2020. http://doi.org/10.3390/sym12020293

- 2019
  - SILVA, Petrônio C. L. Scalable Models of Fuzzy Time Series for Probabilistic Forecasting. PhD Thesis. https://doi.org/10.5281/zenodo.3374641
  - SADAEI, Hossein J. et al. Short-term load forecasting by using a combined method of convolutional neural networks and fuzzy time series. Energy, v. 175, p. 365-377, 2019. http://doi.org/10.1016/j.energy.2019.03.081
  - SILVA, Petrônio CL et al. Probabilistic forecasting with fuzzy time series. IEEE Transactions on Fuzzy Systems, 2019.  http://doi.org/10.1109/TFUZZ.2019.2922152
  - SILVA, Petrônio C. L.; LUCAS, Patrícia de O.; GUIMARÃES, Frederico Gadelha. A Distributed Algorithm for Scalable Fuzzy Time Series. In: International Conference on Green, Pervasive, and Cloud Computing. Springer, Cham, 2019. p. 42-56. http://doi.org/10.1007/978-3-030-19223-5_4 
  - SILVA, Petrônio Cândido de Lima et al. A New Granular Approach for Multivariate Forecasting. In: Latin American Workshop on Computational Neuroscience. Springer, Cham, 2019. p. 41-58. http://doi.org/10.1007/978-3-030-36636-0_4 
  - ALVES, Marcos Antonio et al. Otimizaçao Dinâmica Evolucionária para Despacho de Energia em uma Microrrede usando Veıculos Elétricos. Em: Anais do 14º Simpósio Brasileiro de Automação Inteligente. Campinas : GALOÁ. 2019. http://doi.org/10.17648/sbai-2019-111524
  - LUCAS, Patrícia de O.; SILVA, Petrônio C. L.; GUIMARAES, Frederico G. Otimização Evolutiva de Hiperparâmetros para Modelos de Séries Temporais Nebulosas.Em: Anais do 14º Simpósio Brasileiro de Automação Inteligente. Campinas : GALOÁ. 2019. http://doi.org/10.17648/sbai-2019-111141

- 2018
  - ALVES, Marcos Antônio et al. An extension of nonstationary fuzzy sets to heteroskedastic fuzzy time series. In: ESANN. 2018.

- 2017
  - SEVERIANO, Carlos A. et al. Very short-term solar forecasting using fuzzy time series. In: 2017 IEEE international conference on fuzzy systems (FUZZ-IEEE). IEEE, 2017. p. 1-6. http://doi.org/10.1109/FUZZ-IEEE.2017.8015732
  - SILVA, Petrônio C. L.; et al. Probabilistic forecasting with seasonal ensemble fuzzy time-series. In: XIII Brazilian Congress on Computational Intelligence, Rio de Janeiro. 2017. http://doi.org/10.21528/CBIC2017-54
  - COSTA, Francirley R. B.; SILVA, Petrônio C. L.; GUIMARAES, Frederico G. REGRESSÃO LINEAR APLICADA NA PREDIÇÃO DE SERIES TEMPORAIS FUZZY. Simpósio Brasileiro de Automação Inteligente (SBAI), 2017.

- 2016
  - SILVA, Petrônio C. L.; SADAEI, Hossein Javedani; GUIMARAES, Frederico G. Interval forecasting with fuzzy time series. In: 2016 IEEE Symposium Series on Computational Intelligence (SSCI). IEEE, 2016. p. 1-8. http://doi.org/10.1109/SSCI.2016.7850010
 

## References

1. Q. Song and B. S. Chissom, “Fuzzy time series and its models,” Fuzzy Sets Syst., vol. 54, no. 3, pp. 269–277, 1993.
2. S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
3. C. H. Cheng, R. J. Chang, and C. A. Yeh, “Entropy-based and trapezoidal fuzzification-based fuzzy time series approach for forecasting IT project cost”. Technol. Forecast. Social Change, vol. 73, no. 5, pp. 524–542, Jun. 2006.
4. K. H. Huarng, “Effective lengths of intervals to improve forecasting in fuzzy time series”. Fuzzy Sets Syst., vol. 123, no. 3, pp. 387–394, Nov. 2001.
5. H.-K. Yu, “Weighted fuzzy time series models for TAIEX forecasting”. Phys. A Stat. Mech. its Appl., vol. 349, no. 3, pp. 609–624, 2005.
6. R. Efendi, Z. Ismail, and M. M. Deris, “Improved weight Fuzzy Time Series as used in the exchange rates forecasting of US Dollar to Ringgit Malaysia,” Int. J. Comput. Intell. Appl., vol. 12, no. 1, p. 1350005, 2013.
7. H. J. Sadaei, R. Enayatifar, A. H. Abdullah, and A. Gani, “Short-term load forecasting using a hybrid model with a refined exponentially weighted fuzzy time series and an improved harmony search,” Int. J. Electr. Power Energy Syst., vol. 62, no. from 2005, pp. 118–129, 2014.
8. C.-H. Cheng, Y.-S. Chen, and Y.-L. Wu, “Forecasting innovation diffusion of products using trend-weighted fuzzy time-series model,” Expert Syst. Appl., vol. 36, no. 2, pp. 1826–1832, 2009.
