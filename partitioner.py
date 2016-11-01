import numpy as np
from pyFTS import *

#print(common.__dict__)

def GridPartitionerTrimf(data,npart,names = None,prefix = "A"):
	sets = []
	dmax = max(data)
	dmax = dmax + dmax*0.10
	dmin = min(data)
	dmin = dmin - dmin*0.10
	dlen = dmax - dmin
	partlen = dlen / npart
	partition = dmin
	for c in range(npart):
		sets.append(common.FuzzySet(prefix+str(c),common.trimf,[partition-partlen, partition, partition+partlen], partition ) )
		partition = partition + partlen

	return sets
