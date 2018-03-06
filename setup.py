from distutils.core import setup
from distutils.command.install import install as _install


class install(_install):
    def run(self):
        _install.run(self)

        from pyFTS.data import INMET,Enrollments,AirPassengers,NASDAQ,SONDA,SP500,sunspots,TAIEX
        print("Downloading data assets:")
        print("TAIEX")
        TAIEX.get_data()
        print("sunspots")
        sunspots.get_data()
        print("SP500")
        SP500.get_dataframe()
        print("SONDA")
        SONDA.get_dataframe()
        print("NASDAQ")
        NASDAQ.get_dataframe()
        print("AirPassengers")
        AirPassengers.get_data()
        print("Enrollments")
        Enrollments.get_data()
        print("INMET")
        INMET.get_dataframe()


setup(
    name='pyFTS',
    packages=['pyFTS', 'pyFTS.benchmarks', 'pyFTS.common', 'pyFTS.data', 'pyFTS.models.ensemble',
              'pyFTS.models', 'pyFTS.models.seasonal', 'pyFTS.partitioners', 'pyFTS.probabilistic',
              'pyFTS.tests', 'pyFTS.models.nonstationary', 'pyFTS.models.multivariate'],
    #package_dir={}
    #package_data={'pyFTS.data': ['*.csv','*.csv.bz2']},
    #data_files=[('data', ['pyFTS/data/Enrollments.csv', 'pyFTS/data/AirPassengers.csv'])],
    #include_package_data=True,
    version='1.2.0',
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
    cmdclass={'install': install},
)
