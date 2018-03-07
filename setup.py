from distutils.core import setup

setup(
    name='pyFTS',
    packages=['pyFTS', 'pyFTS.benchmarks', 'pyFTS.common', 'pyFTS.data', 'pyFTS.models.ensemble',
              'pyFTS.models', 'pyFTS.models.seasonal', 'pyFTS.partitioners', 'pyFTS.probabilistic',
              'pyFTS.tests', 'pyFTS.models.nonstationary', 'pyFTS.models.multivariate'],
    #package_dir={}
    #package_data={'pyFTS.data': ['*.csv','*.csv.bz2']},
    #data_files=[('data', ['pyFTS/data/Enrollments.csv', 'pyFTS/data/AirPassengers.csv'])],
    #include_package_data=True,
    version='1.2.2',
    description='Fuzzy Time Series for Python',
    author='Petronio Candido L. e Silva',
    author_email='petronio.candido@gmail.com',
    url='https://github.com/petroniocandido/pyFTS',
    download_url='https://github.com/petroniocandido/pyFTS/archive/pkg1.2.3.tar.gz',
    keywords=['forecasting', 'fuzzy time series', 'fuzzy', 'time series forecasting'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
