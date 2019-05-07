from distutils.core import setup

setup(
    name='pyFTS',
    packages=['pyFTS', 'pyFTS.benchmarks', 'pyFTS.common', 'pyFTS.data', 'pyFTS.models.ensemble',
              'pyFTS.models', 'pyFTS.models.seasonal', 'pyFTS.partitioners', 'pyFTS.probabilistic',
              'pyFTS.tests', 'pyFTS.models.nonstationary', 'pyFTS.models.multivariate',
              'pyFTS.models.incremental', 'pyFTS.hyperparam', 'pyFTS.distributed', 'pyFTS.fcm'],
    version='1.6',
    description='Fuzzy Time Series for Python',
    author='Petronio Candido L. e Silva',
    author_email='petronio.candido@gmail.com',
    url='https://pyfts.github.io/pyFTS/',
    download_url='https://github.com/PYFTS/pyFTS/archive/pkg1.6.tar.gz',
    keywords=['forecasting', 'fuzzy time series', 'fuzzy', 'time series forecasting'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
