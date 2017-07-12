from distutils.core import setup
setup(
  name = 'pyFTS',
  packages = ['pyFTS','pyFTS.benchmarks','pyFTS.common','pyFTS.data', 'pyFTS.ensemble','pyFTS.models','pyFTS.models.seasonal','pyFTS.partitioners','pyFTS.probabilistic','pyFTS.tests'], 
  package_data = {'benchmarks':['*'], 'common':['*'], 'data':['*'], 'ensemble':['*'], 'models':['*'], 'seasonal':['*'], 'partitioners':['*'], 'probabilistic':['*'], 'tests':['*']},
  version = '1.2',
  description = 'Fuzzy Time Series for Python',
  author = 'Petronio Candido L. e Silva',
  author_email = 'petronio.candido@gmail.com',
  url = 'https://github.com/petroniocandido/pyFTS',
  download_url = 'https://github.com/petroniocandido/pyFTS/archive/pkg1.2.tar.gz',
  keywords = ['forecasting', 'fuzzy time series', 'fuzzy', 'time series forecasting'],
  classifiers = [],
)
