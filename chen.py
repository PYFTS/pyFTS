import numpy as np
from pyFTS import *

class ConventionalFLRG:
	def __init__(self,LHS):
		self.LHS = LHS
		self.RHS = set()
	
	def append(self,c):
		self.RHS.add(c)

	def __str__(self):
		tmp = self.LHS + " -> "
		tmp2 = ""
		for c in self.RHS:
			if len(tmp2) > 0:
				tmp2 = tmp2 + ","
			tmp2 = tmp2 + c
		return tmp + tmp2


class ConventionalFTS(fts.FTS):
	def __init__(self,name):
		super(ConventionalFTS, self).__init__(1,name)
		self.flrgs = {}
    
    def generateFLRG(self, flrs):
		flrgs = {}
		for flr in flrs:
			if flr.LHS in flrgs:
				flrgs[flr.LHS].append(flr.RHS)
			else:
				flrgs[flr.LHS] = ConventionalFLRG(flr.LHS);
				flrgs[flr.LHS].append(flr.RHS)
		return (flrgs)

	def train(self, data, sets):
		self.sets = sets
		tmpdata = common.fuzzySeries(data,sets)
		flrs = common.generateNonRecurrentFLRs(tmpdata)
		self.flrgs = self.generateFLRG(flrs)
        
	def forecast(self,data):
		
		mv = common.fuzzyInstance(data, self.sets)
		
		actual = self.sets[ np.argwhere( mv == max(mv) )[0,0] ]
        
		if actual.name not in self.flrgs:
			return actual.centroid

		flrg = self.flrgs[actual.name]

		count = 0.0
		denom = 0.0

		for s in flrg.RHS:
			denom = denom + self.sets[s].centroid
			count = count + 1.0

		return denom/count
		
	
		 
