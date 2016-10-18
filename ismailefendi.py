import numpy as np
from pyFTS import *

class ImprovedWeightedFLRG:
	def __init__(self,LHS):
		self.LHS = LHS
		self.RHS = {}
		self.count = 0.0

	def append(self,c):
		if c not in self.RHS:
			self.RHS[c] = 1.0
		else:
			self.RHS[c] = self.RHS[c] + 1.0
		self.count = self.count + 1.0

	def weights(self):
		return np.array([ self.RHS[c]/self.count for c in self.RHS.keys() ])
        
	def __str__(self):
		tmp = self.LHS + " -> "
		tmp2 = ""
		for c in self.RHS.keys():
			if len(tmp2) > 0:
				tmp2 = tmp2 + ","
			tmp2 = tmp2 + c + "(" + str(round(self.RHS[c]/self.count,3)) + ")"
		return tmp + tmp2
		

class ImprovedWeightedFTS(fts.FTS):
	def __init__(self,name):
		super(ImprovedWeightedFTS, self).__init__(1,name)
		
	def generateFLRG(self, flrs):
		flrgs = {}
		for flr in flrs:
			if flr.LHS in flrgs:
				flrgs[flr.LHS].append(flr.RHS)
			else:
				flrgs[flr.LHS] = ImprovedWeightedFLRG(flr.LHS);
				flrgs[flr.LHS].append(flr.RHS)
		return (flrgs)

	def train(self, data, sets):
		self.sets = sets
		tmpdata = common.fuzzySeries(data,sets)
		flrs = common.generateRecurrentFLRs(tmpdata)
		self.flrgs = self.generateFLRG(flrs)
        
	def forecast(self,data):
		mv = common.fuzzyInstance(data, self.sets)
		
		actual = self.sets[ np.argwhere( mv == max(mv) )[0,0] ]
        
		if actual.name not in self.flrgs:
			return actual.centroid

		flrg = self.flrgs[actual.name]
		
		mi = np.array([self.sets[s].centroid for s in flrg.RHS.keys()])
		return mi.dot( flrg.weights() )
