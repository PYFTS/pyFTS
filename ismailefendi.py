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
        
	def defuzzy(self,data):
		actual = self.fuzzy(data)
		if actual["fuzzyset"] not in self.flrgs:
			return self.sets[actual["fuzzyset"]].centroid
		flrg = self.flrgs[actual["fuzzyset"]]
		mi = np.array([self.sets[s].centroid for s in flrg.RHS.keys()])
		return mi.dot( flrg.weights() )
        
	def learn(self, data, sets):
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
					self.flrgs[last["fuzzyset"]] = ImprovedWeightedFLRG(last["fuzzyset"])
			
				self.flrgs[last["fuzzyset"]].append(actual["fuzzyset"])    
			count = count + 1
			last = actual
