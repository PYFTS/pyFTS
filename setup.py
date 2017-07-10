from distutils.core import setup
setup(
  name = 'pyFTS',
  packages = ['pyFTS','pyFTS.benchmarks','pyFTS.common','pyFTS.data', 'pyFTS.ensemble','pyFTS.models','pyFTS.partitioners','pyFTS.probabilistic','pyFTS.tests'], 
  package_data = {'benchmarks':['*'], 'common':['*'], 'data':['*'], 'ensemble':['*'], 'models':['*'], 'partitioners':['*'], 'probabilistic':['*'], 'tests':['*']},
  version = '0.1',
  description = 'Fuzzy Time Series for Python',
  author = 'Petronio Candido L. e Silva',
  author_email = 'petronio.candido@gmail.com',
  url = 'https://github.com/petroniocandido/pyFTS',
  download_url = 'https://github.com/petroniocandido/pyFTS/archive/pkg0.1.tar.gz',
  keywords = ['forecasting', 'fuzzy time series', 'fuzzy', 'time series forecasting'],
  classifiers = [],
)
