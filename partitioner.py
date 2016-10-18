import numpy as np
from pyFTS import *

#print(common.__dict__)

def GridPartitionerTrimf(data,npart,names = None,prefix = "A"):
	sets = []
	dmax = max(data)
	dmin = min(data)
	dlen = dmax - dmin
	partlen = dlen / npart
	partition = dmin
	for c in range(npart):
		sets.append(common.FuzzySet(prefix+str(c),common.trimf,[partition-partlen, partition, partition+partlen], partition ) )
		partition = partition + partlen

	return sets
