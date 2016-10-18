import numpy as np
from pyFTS import *

class ExponentialyWeightedFLRG:
	def __init__(self,LHS,c):
		self.LHS = LHS
		self.RHS = []
		self.count = 0.0
		self.c = c

	def append(self,c):
		self.RHS.append(c)
		self.count = self.count + 1.0

	def weights(self):
		wei = [ self.c**k for k in np.arange(0.0,self.count,1.0)]
		tot = sum( wei )
		return np.array([ k/tot for k in wei ])
        
	def __str__(self):
		tmp = self.LHS + " -> "
		tmp2 = ""
		cc = 0
		wei = [ self.c**k for k in np.arange(0.0,self.count,1.0)]
		tot = sum( wei )
		for c in self.RHS:
			if len(tmp2) > 0:
				tmp2 = tmp2 + ","
			tmp2 = tmp2 + c + "(" + str(wei[cc]/tot) + ")"
			cc = cc + 1
		return tmp + tmp2
		
class ExponentialyWeightedFTS(fts.FTS):
	def __init__(self,name):
		super(ExponentialyWeightedFTS, self).__init__(1,name)
		this.c = 1
		
	def generateFLRG(self, flrs, c):
		flrgs = {}
		for flr in flrs:
			if flr.LHS in flrgs:
				flrgs[flr.LHS].append(flr.RHS)
			else:
				flrgs[flr.LHS] = ExponentialyWeightedFLRG(flr.LHS, c);
				flrgs[flr.LHS].append(flr.RHS)
		return (flrgs)

	def train(self, data, sets, c):
		this.c = c
		self.sets = sets
		tmpdata = common.fuzzySeries(data,sets)
		flrs = common.generateRecurrentFLRs(tmpdata)
		self.flrgs = self.generateFLRG(flrs,c)
        
	def forecast(self,data):
        mv = common.fuzzyInstance(data, self.sets)
		
		actual = self.sets[ np.argwhere( mv == max(mv) )[0,0] ]
        
		if actual.name not in self.flrgs:
			return actual.centroid

		flrg = self.flrgs[actual.name]

		mi = np.array([self.sets[s].centroid for s in flrg.RHS])
        
		return mi.dot( flrg.weights() )
        
