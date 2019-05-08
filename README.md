# pyFTS - Fuzzy Time Series for Python

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/pyFTS/)
[![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

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

1. Alves, M. A., Silva, P. C. L., Severiano, C. A. J., Vieira, G. L., Guimaraes, F. G., Sadaei, H. J. An extension of nonstationary fuzzy sets to heteroskedastic fuzzy time series. 26th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, 2018.
2. Silva, P. C. L., Alves, M. A., Alberto, C., Junior, S., Vieira, G. L., Guimaraes, F. G.; Sadaei, H. J. Probabilistic Forecasting with Seasonal Ensemble Fuzzy Time-Series. XIII Brazilian Congress on Computational Intelligence, 2017.
3. Severiano, S. A. Jr, Silva, P. C. L., Sadaei, H. J., Guimaraes, F. G. Very Short-term Solar Forecasting using Fuzzy Time Series. 2017 IEEE International Conference on Fuzzy Systems. DOI10.1109/FUZZ-IEEE.2017.8015732
4. Silva, P. C. L., Sadaei, H. J., Guimaraes, F. G. Interval Forecasting with Fuzzy Time Series. In: Computational Intelligence (SSCI), 2016 IEEE Symposium Series on. IEEE, 2016. p. 1-8.
 

## References

1. Q. Song and B. S. Chissom, “Fuzzy time series and its models,” Fuzzy Sets Syst., vol. 54, no. 3, pp. 269–277, 1993.
2. S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
3. C. H. Cheng, R. J. Chang, and C. A. Yeh, “Entropy-based and trapezoidal fuzzification-based fuzzy time series approach for forecasting IT project cost”. Technol. Forecast. Social Change, vol. 73, no. 5, pp. 524–542, Jun. 2006.
4. K. H. Huarng, “Effective lengths of intervals to improve forecasting in fuzzy time series”. Fuzzy Sets Syst., vol. 123, no. 3, pp. 387–394, Nov. 2001.
5. H.-K. Yu, “Weighted fuzzy time series models for TAIEX forecasting”. Phys. A Stat. Mech. its Appl., vol. 349, no. 3, pp. 609–624, 2005.
6. R. Efendi, Z. Ismail, and M. M. Deris, “Improved weight Fuzzy Time Series as used in the exchange rates forecasting of US Dollar to Ringgit Malaysia,” Int. J. Comput. Intell. Appl., vol. 12, no. 1, p. 1350005, 2013.
7. H. J. Sadaei, R. Enayatifar, A. H. Abdullah, and A. Gani, “Short-term load forecasting using a hybrid model with a refined exponentially weighted fuzzy time series and an improved harmony search,” Int. J. Electr. Power Energy Syst., vol. 62, no. from 2005, pp. 118–129, 2014.
8. C.-H. Cheng, Y.-S. Chen, and Y.-L. Wu, “Forecasting innovation diffusion of products using trend-weighted fuzzy time-series model,” Expert Syst. Appl., vol. 36, no. 2, pp. 1826–1832, 2009.
