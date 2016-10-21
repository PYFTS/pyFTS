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
		
	def buildTree(self,node, lags, level):
		if level >= self.order:
			return
			
		for s in lags[level]["sets"]:				
			node.appendChild(tree.FLRGTreeNode(s))
			
		for child in node.getChildren():
			self.buildTree(child,lags,level+1)
    	
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
				count = count + 1				
				
			# Constrói uma árvore com todos os caminhos possíveis
			
			root = tree.FLRGTreeNode(None)
			
			self.buildTree(root,lags,0)
			
			# Traça os possíveis caminhos e costróis as HOFLRG's
			
			print(root)
			
			# -------
				
			for p in root.paths():
				path = list(reversed(list(filter(None.__ne__, p))))
				print(path)
				#for n in tree.flat(p):
				#	print(n)
				#print("--")
					
			return
				
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
		
	
		 
