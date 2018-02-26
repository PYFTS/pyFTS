from distutils.core import setup

setup(
    name='pyFTS',
    packages=['pyFTS', 'pyFTS.benchmarks', 'pyFTS.common', 'pyFTS.data', 'pyFTS.models.ensemble',
              'pyFTS.models', 'pyFTS.models.seasonal', 'pyFTS.partitioners', 'pyFTS.probabilistic',
              'pyFTS.tests', 'pyFTS.models.nonstationary'],
    #package_dir={}
    package_data={'benchmarks': ['*'], 'common': ['*'], 'data': ['*'],
                  'models': ['*'], 'seasonal': ['*'], 'ensemble': ['*'],
                  'partitioners': ['*'], 'probabilistic': ['*'], 'tests': ['*']},
    #data_files=[('data', ['pyFTS/data/Enrollments.csv', 'pyFTS/data/AirPassengers.csv'])],
    include_package_data=True,
    version='1.1.1',
    description='Fuzzy Time Series for Python',
    author='Petronio Candido L. e Silva',
    author_email='petronio.candido@gmail.com',
    url='https://github.com/petroniocandido/pyFTS',
    download_url='https://github.com/petroniocandido/pyFTS/archive/pkg1.1.1.tar.gz',
    keywords=['forecasting', 'fuzzy time series', 'fuzzy', 'time series forecasting'],
    classifiers=[],
    #install_requires=[
    #    'numpy','pandas','matplotlib','dill','copy','dispy','multiprocessing','joblib'
    #],
)
