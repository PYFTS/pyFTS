import numpy as np
from pyFTS import *

class ImprovedWeightedFLRG:
	def __init__(self,LHS):
		self.LHS = LHS
		self.RHS = {}
		self.count = 0.0

	def append(self,c):
		if c.name not in self.RHS:
			self.RHS[c.name] = 1.0
		else:
			self.RHS[c.name] = self.RHS[c.name] + 1.0
		self.count = self.count + 1.0

	def weights(self):
		return np.array([ self.RHS[c]/self.count for c in self.RHS.keys() ])
        
	def __str__(self):
		tmp = self.LHS.name + " -> "
		tmp2 = ""
		for c in sorted(self.RHS):
			if len(tmp2) > 0:
				tmp2 = tmp2 + ","
			tmp2 = tmp2 + c + "(" + str(round(self.RHS[c]/self.count,3)) + ")"
		return tmp + tmp2
		

class ImprovedWeightedFTS(fts.FTS):
	def __init__(self,name):
		super(ImprovedWeightedFTS, self).__init__(1,name)
		self.setsDict = {}
		
	def generateFLRG(self, flrs):
		flrgs = {}
		for flr in flrs:
			if flr.LHS.name in flrgs:
				flrgs[flr.LHS.name].append(flr.RHS)
			else:
				flrgs[flr.LHS.name] = ImprovedWeightedFLRG(flr.LHS);
				flrgs[flr.LHS.name].append(flr.RHS)
		return (flrgs)

	def train(self, data, sets):
		self.sets = sets
		
		for s in self.sets:	self.setsDict[s.name] = s
		
		tmpdata = common.fuzzySeries(data,self.sets)
		flrs = common.generateRecurrentFLRs(tmpdata)
		self.flrgs = self.generateFLRG(flrs)
		
	def getMidpoints(self,flrg):
		ret = np.array([self.setsDict[s].centroid for s in flrg.RHS])
		return ret
        
	def forecast(self,data):
		l = 1
		
		ndata = np.array(data)
		
		l = len(ndata)
		
		ret = []
		
		for k in np.arange(1,l):
			
			mv = common.fuzzyInstance(ndata[k], self.sets)
		
			actual = self.sets[ np.argwhere( mv == max(mv) )[0,0] ]
        
			if actual.name not in self.flrgs:
				ret.append(actual.centroid)
			else:
				flrg = self.flrgs[actual.name]
				mp = self.getMidpoints(flrg)
				
				ret.append( mp.dot( flrg.weights() ))
			
		return ret
