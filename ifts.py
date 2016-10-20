import numpy as np
from pyFTS import *


class IntervalFTS(hofts.HighOrderFTS):
	def __init__(self,name):
		super(IntervalFTS, self).__init__(name)
		self.flrgs = {}
    
	def getUpper(self,flrg):
		ret = np.array([s.upper for s in flrg.RHS])
		return ret
		
	def getLower(self,flrg):
		ret = np.array([s.lower for s in flrg.RHS])
		return ret
		
	def getSequenceMembership(self, data, fuzzySets):
		mb = [ fuzzySets[k].membership( data[k] )  for k in np.arange(0,len(data))  ]
		return mb
    	
	def forecast(self,data):
		
		ndata = np.array(data)
		
		l = len(ndata)
		
		ret = []
		
		for k in np.arange(self.order+1,l):
			
			up = []
			lo = []
			
			# Achar os conjuntos que tem pert > 0 para cada lag
			count = 0
			lags = {}
			combinations = 1
			for instance in ndata[k-self.order : k]:
				mb = common.fuzzyInstance(instance, self.sets)
				tmp = np.argwhere( mb )
				idx = np.ravel(tmp) #flatten 
				
				lag = {}
				lag["id"] = count
				lag["sets"] = idx
				lag["memberships"] = [mb[ k ] for k in idx]
				lag["count"] = len(idx)
				lags[count] = lag
				combinations = combinations * lag["count"]
				count = count + 1				
				
				print(combinations)
				
				
			# Build a tree exploring all possibilities
			
			# Trace each path from leaf to roots and reverse path
			
			# -------
				
			#return lags
				
			wflrgs = {}
			
			# Gerar as permutações possíveis e as FLRG's correspondentes
			lag_inc = [0 for k in np.arange(0,self.order) ]
			isComplete = False
			while (isComplete):
				flrg = hofts.HighOrderFLRG(self.order)
				
				flrg.appendLHS( self.sets[ lag_inc[0]  ] )
				
				for lag_count in np.arange(1,self.order):
					if lag_count > 1: lag_inc[ lag_count - 1 ] = 0
					
#					for 
					
				#lag_count = lag_count + 1
			
			# Achar a pert geral de cada FLRG
			
			# achar o os bounds de cada FLRG
			
			# gerar o intervalo
			
#			tmpdata = common.fuzzySeries(XX,self.sets)
#			tmpflrg = HighOrderFLRG(self.order)
		
#			for s in tmpdata: tmpflrg.appendLHS(s)
			
#			if tmpflrg.strLHS() not in self.flrgs:
#				ret.append(tmpdata[-1].centroid)
#			else:
#				flrg = self.flrgs[tmpflrg.strLHS()]
#				mp = self.getMidpoints(flrg)
				
#				ret.append(sum(mp)/len(mp))
			
#		return ret
		
	
		 
