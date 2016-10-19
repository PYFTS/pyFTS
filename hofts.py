import numpy as np
from pyFTS import *

class HighOrderFLRG:
	def __init__(self,order):
		self.LHS = []
		self.RHS = []
		self.order = order
		self.strlhs = ""

	def appendRHS(self,c):
		self.RHS.append(c)
		
	def strLHS(self):
		if len(self.strlhs) == 0:
			for c in self.LHS:
				if len(self.strlhs) > 0:
					self.strlhs = self.strlhs + ","
				self.strlhs = self.strlhs + c.name
		return self.strlhs
	
	def appendLHS(self,c):
		self.LHS.append(c)

	def __str__(self):
		tmp = ""
		for c in sorted(self.RHS, key=lambda s: s.name):
			if len(tmp) > 0:
				tmp = tmp + ","
			tmp = tmp + c.name
		return self.strLHS() + " -> " + tmp
		
class HighOrderFTS(fts.FTS):
	def __init__(self,name):
		super(HighOrderFTS, self).__init__(1,name)
		self.order = 1
		
	def generateFLRG(self, flrs):
		flrgs = {}
		l = len(flrs)
		for k in np.arange(self.order +1, l):
			flrg = HighOrderFLRG(self.order)
			
			for kk in np.arange(k - self.order, k):
				flrg.appendLHS( flrs[kk].LHS )
						
			if flrg.strLHS() in flrgs:
				flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
			else:
				flrgs[flrg.strLHS()] = flrg;
				flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
		return (flrgs)

	def train(self, data, sets, order):
		self.order = order
		self.sets = sets
		tmpdata = common.fuzzySeries(data,sets)
		flrs = common.generateRecurrentFLRs(tmpdata)
		self.flrgs = self.generateFLRG(flrs)
        
	def forecast(self,data):
		
		ret = []
		
		l = len(data)
		
		if l <= self.order:
			return data
			
		for k in np.arange(self.order, l):
			tmpdata = common.fuzzySeries(data[k-self.order : k],self.sets)
			tmpflrg = HighOrderFLRG(self.order)
		
			for s in tmpdata: tmpflrg.appendLHS(s)
			
			if tmpflrg.strLHS() not in self.flrgs:
				ret.append(tmpdata[-1].centroid)
			else:
				flrg = self.flrgs[tmpflrg.strLHS()]
				mp = self.getMidpoints(flrg)
				
				ret.append(sum(mp)/len(mp))
			
		return ret
		
