# pyFTS - Fuzzy Time Series for Python

## pyFTS Library

This package is intended for students, researchers, data scientists or whose want to exploit the Fuzzy Time Series methods. These methods provide simple, easy to use, computationally cheap and human-readable models, suitable for statistic laymans to experts.

This project is continously under improvement and contributors are well come.


## Fuzzy Time Series (FTS)
Fuzzy Time Series (FTS) are non parametric methods for time series forecasting based on Fuzzy Theory.  The original method was proposed by [1] and improved later by many researchers. The general approach of the FTS methods, based on [2] is listed below:

1. **Data preprocessing**: Data transformation functions contained at pyFTS.common.Transformations, like differentiation, Box-Cox, scaling and normalization.
2. **Universe of Discourse Partitioning**: This is the most important step. Here, the range of values of the numerical time series *Y(t)* will be splited in overlapped intervals and for each interval will be created a Fuzzy Set. This step is performed by pyFTS.partition module and its classes (for instance GridPartitioner, EntropyPartitioner, etc). The main parameters are:
 - the number of intervals
 - which fuzzy membership function (on pyFTS.common.Membership)
 - partition scheme (GridPartitioner, EntropyPartitioner, FCMPartitioner, CMeansPartitioner, HuarngPartitioner)
 
 Check the jupyter notebook /notebooks/Partitioners.ipynb for sample codes.
 
3. **Data Fuzzyfication**: Each data point of the numerical time series *Y(t)* will be translated to a fuzzy representation (usually one or more fuzzy sets), and then a fuzzy time series *F(t)* is created.

4. **Generation of Fuzzy Rules**: In this step the temporal transition rules are created. These rules depends on the method and their characteristics:
- *order*: the number of time lags used on forecasting
- *weights*: the weighted models introduce weights on fuzzy rules for smoothing
- *seasonality*: seasonality models depends 
- *steps ahead*: the number of steps ahed to predict. Almost all standard methods are based on one-step-ahead forecasting
- *forecasting type*: Almost all standard methods are point-based, but pyFTS also provides intervalar and probabilistic forecasting methods.

5. **Forecasting**: 

6. **Defuzzyfication**

7. **Data postprocessing**: The inverse operations of step 1.

