pyFTS Quick Start
=================

How to install pyFTS?
---------------------

.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg

Before of all, pyFTS was developed and tested with Python 3.6. To install pyFTS using pip tool

	pip install -U pyFTS

Ou clone directly from the GitHub repo for the most recent review:

	pip install -U git+https://github.com/PYFTS/pyFTS


What are Fuzzy Time Series (FTS)?
---------------------------------

Fuzzy Time Series (FTS) are non parametric methods for time series forecasting based on Fuzzy Theory.  The original method was proposed by [1] and improved later by many researchers. The general approach of the FTS methods, based on [2] is listed below:

1. **Data preprocessing**: Data transformation functions contained at `pyFTS.common.Transformations <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/common/Transformations.py>`_, like differentiation, Box-Cox, scaling and normalization.

2. **Universe of Discourse Partitioning**: This is the most important step. Here, the range of values of the numerical time series *Y(t)* will be splited in overlapped intervals and for each interval will be created a Fuzzy Set. This step is performed by pyFTS.partition module and its classes (for instance GridPartitioner, EntropyPartitioner, etc). The main parameters are:
 - the number of intervals
 - which fuzzy membership function (on `pyFTS.common.Membership <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/common/Membership.py>`_)
 - partition scheme (`GridPartitioner <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/Grid.py>`_, `EntropyPartitioner <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/Entropy.py>`_, `FCMPartitioner <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/FCM.py>`_, `CMeansPartitioner <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/CMeans.py>`_, `HuarngPartitioner <https://github.com/PYFTS/pyFTS/blob/master/pyFTS/partitioners/Huarng.py>`_)
 
 Check out the jupyter notebook on `pyFTS/notebooks/Partitioners.ipynb <https://github.com/PYFTS/notebooks/Partitioners.ipynb>`_ for sample codes.
 
3. **Data Fuzzyfication**: Each data point of the numerical time series *Y(t)* will be translated to a fuzzy representation (usually one or more fuzzy sets), and then a fuzzy time series *F(t)* is created.

4. **Generation of Fuzzy Rules**: In this step the temporal transition rules are created. These rules depends on the method and their characteristics:
- *order*: the number of time lags used on forecasting
- *weights*: the weighted models introduce weights on fuzzy rules for smoothing
- *seasonality*: seasonality models 
- *steps ahead*: the number of steps ahed to predict. Almost all standard methods are based on one-step-ahead forecasting
- *forecasting type*: Almost all standard methods are point-based, but pyFTS also provides intervalar and probabilistic forecasting methods.

5. **Forecasting**: The forecasting step takes a sample (with minimum length equal to the model's order) and generate a fuzzy outputs (fuzzy set(s)) for the next time ahead. 

6. **Defuzzyfication**: This step transform the fuzzy forecast into a real number.

7. **Data postprocessing**: The inverse operations of step 1.

Usage examples
--------------

There is nothing better than good code examples to start. `Then check out the demo Jupyter Notebooks of the implemented method os pyFTS! <https://github.com/PYFTS/notebooks>`_.

A Google Colab example can also be found `here <https://drive.google.com/file/d/1zRBCHXOawwgmzjEoKBgmvBqkIrKxuaz9/view?usp=sharing>`_.

References
----------

1. Q. Song and B. S. Chissom, “Fuzzy time series and its models,” Fuzzy Sets Syst., vol. 54, no. 3, pp. 269–277, 1993.
2. S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
3. C. H. Cheng, R. J. Chang, and C. A. Yeh, “Entropy-based and trapezoidal fuzzification-based fuzzy time series approach for forecasting IT project cost”. Technol. Forecast. Social Change, vol. 73, no. 5, pp. 524–542, Jun. 2006.
4. K. H. Huarng, “Effective lengths of intervals to improve forecasting in fuzzy time series”. Fuzzy Sets Syst., vol. 123, no. 3, pp. 387–394, Nov. 2001.
5. H.-K. Yu, “Weighted fuzzy time series models for TAIEX forecasting”. Phys. A Stat. Mech. its Appl., vol. 349, no. 3, pp. 609–624, 2005.
6. R. Efendi, Z. Ismail, and M. M. Deris, “Improved weight Fuzzy Time Series as used in the exchange rates forecasting of US Dollar to Ringgit Malaysia,” Int. J. Comput. Intell. Appl., vol. 12, no. 1, p. 1350005, 2013.
7. H. J. Sadaei, R. Enayatifar, A. H. Abdullah, and A. Gani, “Short-term load forecasting using a hybrid model with a refined exponentially weighted fuzzy time series and an improved harmony search,” Int. J. Electr. Power Energy Syst., vol. 62, no. from 2005, pp. 118–129, 2014.
8. C.-H. Cheng, Y.-S. Chen, and Y.-L. Wu, “Forecasting innovation diffusion of products using trend-weighted fuzzy time-series model,” Expert Syst. Appl., vol. 36, no. 2, pp. 1826–1832, 2009.
