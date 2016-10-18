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
        
	def forecast(self,data):
        
		actual = self.fuzzy(data)
        
		if actual["fuzzyset"] not in self.flrgs:
			return self.sets[actual["fuzzyset"]].centroid

		flrg = self.flrgs[actual["fuzzyset"]]

		mi = np.array([self.sets[s].centroid for s in flrg.RHS])
        
		return mi.dot( flrg.weights() )
        
	def train(self, data, sets):
		last = {"fuzzyset":"", "membership":0.0}
		actual = {"fuzzyset":"", "membership":0.0}
		
		for s in sets:
			self.sets[s.name] = s
		
		self.flrgs = {}
		count = 1
		for inst in data:
			actual = self.fuzzy(inst)
			
			if count > self.order:
				if last["fuzzyset"] not in self.flrgs:
					self.flrgs[last["fuzzyset"]] = ExponentialyWeightedFLRG(last["fuzzyset"],2)
			
				self.flrgs[last["fuzzyset"]].append(actual["fuzzyset"])    
			count = count + 1
			last = actual
