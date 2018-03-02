# pyFTS - Fuzzy Time Series for Python

## pyFTS Library

This package is intended for students, researchers, data scientists or whose want to exploit the Fuzzy Time Series methods. These methods provide simple, easy to use, computationally cheap and human-readable models, suitable for statistic laymans to experts.

This project is continously under improvement and contributors are well come.


## Fuzzy Time Series (FTS)
Fuzzy Time Series (FTS) are non parametric methods for time series forecasting based on Fuzzy Theory.  The original method was proposed by [1] and improved later by many researchers. The general approach of the FTS methods, based on [2] is listed below:

1. **Data preprocessing**: Data transformation functions contained at [pyFTS.common.Transformations](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/common/Transformations.py), like differentiation, Box-Cox, scaling and normalization.

2. **Universe of Discourse Partitioning**: This is the most important step. Here, the range of values of the numerical time series *Y(t)* will be splited in overlapped intervals and for each interval will be created a Fuzzy Set. This step is performed by pyFTS.partition module and its classes (for instance GridPartitioner, EntropyPartitioner, etc). The main parameters are:
 - the number of intervals
 - which fuzzy membership function (on [pyFTS.common.Membership](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/common/Membership.py))
 - partition scheme ([GridPartitioner](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/partitioners/Grid.py), [EntropyPartitioner](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/partitioners/Entropy.py), [FCMPartitioner](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/partitioners/FCM.py), [CMeansPartitioner](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/partitioners/CMeans.py), [HuarngPartitioner](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/partitioners/Huarng.py))
 
 Check out the jupyter notebook on [pyFTS/notebooks/Partitioners.ipynb](https://github.com/petroniocandido/pyFTS/blob/master/pyFTS/notebooks/Partitioners.ipynb) for sample codes.
 
3. **Data Fuzzyfication**: Each data point of the numerical time series *Y(t)* will be translated to a fuzzy representation (usually one or more fuzzy sets), and then a fuzzy time series *F(t)* is created.

4. **Generation of Fuzzy Rules**: In this step the temporal transition rules are created. These rules depends on the method and their characteristics:
- *order*: the number of time lags used on forecasting
- *weights*: the weighted models introduce weights on fuzzy rules for smoothing
- *seasonality*: seasonality models depends 
- *steps ahead*: the number of steps ahed to predict. Almost all standard methods are based on one-step-ahead forecasting
- *forecasting type*: Almost all standard methods are point-based, but pyFTS also provides intervalar and probabilistic forecasting methods.

5. **Forecasting**: The forecasting step takes a sample (with minimum length equal to the model's order) and generate a fuzzy outputs (fuzzy set(s)) for the next time ahead. 

6. **Defuzzyfication**: This step transform the fuzzy forecast into a real number.

7. **Data postprocessing**: The inverse operations of step 1.

