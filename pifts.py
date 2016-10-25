import numpy as np
from pyFTS import *

class ProbabilisticIntervalFLRG(hofts.HighOrderFLRG):
	def __init__(self,order):
		super(ProbabilisticIntervalFLRG, self).__init__(order)
		self.RHS = {}
		self.frequencyCount = 0
		
	def appendRHS(self,c):
		self.frequencyCount = self.frequencyCount + 1
		if c.name in self.RHS:
			self.RHS[c.name] = self.RHS[c.name] + 1
		else:
			self.RHS[c.name] = 1
			
	def getProbability(self,c):
		return self.RHS[c] / self.frequencyCount
		
	def __str__(self):
		tmp2 = ""
		for c in sorted(self.RHS):
			if len(tmp2) > 0:
				tmp2 = tmp2 + ","
			tmp2 = tmp2 + c + "(" + str(round(self.RHS[c]/self.frequencyCount,3)) + ")"
		return self.strLHS() + " -> " + tmp2

class ProbabilisticIntervalFTS(ifts.IntervalFTS):
	def __init__(self,name):
		super(ProbabilisticIntervalFTS, self).__init__(name)
		self.flrgs = {}
		self.globalFrequency = 0
		
	def generateFLRG(self, flrs):
		flrgs = {}
		l = len(flrs)
		for k in np.arange(self.order +1, l):
			flrg = ProbabilisticIntervalFLRG(self.order)
			
			for kk in np.arange(k - self.order, k):
				flrg.appendLHS( flrs[kk].LHS )
						
			if flrg.strLHS() in flrgs:
				flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
			else:
				flrgs[flrg.strLHS()] = flrg;
				flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
				
			self.globalFrequency = self.globalFrequency + 1
		return (flrgs)
		
	def getProbability(self, flrg):
		return flrg.frequencyCount / self.globalFrequency
		
	def getUpper(self,flrg):
		if flrg.strLHS() in self.flrgs:
			tmp = self.flrgs[ flrg.strLHS() ]
			ret = sum(np.array([ tmp.getProbability(s) * self.setsDict[s].upper for s in tmp.RHS]))
		else:
			ret = flrg.LHS[-1].upper
		return ret
		
	def getLower(self,flrg):
		if flrg.strLHS() in self.flrgs:
			tmp = self.flrgs[ flrg.strLHS() ]
			ret = sum(np.array([ tmp.getProbability(s) * self.setsDict[s].lower for s in tmp.RHS]))
		else:
			ret = flrg.LHS[-1].lower
		return ret
    	
	def forecast(self,data):
		
		ndata = np.array(data)
		
		l = len(ndata)
		
		ret = []
		
		for k in np.arange(self.order,l):
			
			print(k)
			
			flrs = []
			mvs = []
			
			up = []
			lo = []
			
			# Achar os conjuntos que tem pert > 0 para cada lag
			count = 0
			lags = {}
			if self.order > 1:
				subset = ndata[k-self.order : k ]
				print(subset)
				for instance in subset:
					mb = common.fuzzyInstance(instance, self.sets)
					tmp = np.argwhere( mb )
					idx = np.ravel(tmp) #flat the array
					lags[count] = idx 
					count = count + 1				
					
				# Constrói uma árvore com todos os caminhos possíveis
				
				root = tree.FLRGTreeNode(None)
				
				self.buildTree(root,lags,0)
				
				# Traça os possíveis caminhos e costrói as HOFLRG's
				
				for p in root.paths():
					path = list(reversed(list(filter(None.__ne__, p))))
					flrg = hofts.HighOrderFLRG(self.order)
					for kk in path: flrg.appendLHS(self.sets[ kk ])
					
					##
					flrs.append( self.flrgs[ flrg.strLHS() ]  )
					
					# Acha a pertinência geral de cada FLRG
					mvs.append(min(self.getSequenceMembership(subset, flrg.LHS)))
			else:
				
				mv = common.fuzzyInstance(ndata[k],self.sets) # get all membership values
				tmp = np.argwhere( mv ) # get the indices of values > 0
				idx = np.ravel(tmp) # flatten the array
				for kk in idx:
					flrg = hofts.HighOrderFLRG(self.order)
					flrg.appendLHS(self.sets[ kk ])
					flrs.append( self.flrgs[ flrg.strLHS() ]  )
					mvs.append(mv[kk])
			
			count = 0
			for flrg in flrs:
				# achar o os bounds de cada FLRG, ponderados pela pertinência
				up.append( self.getProbability(flrg) * mvs[count] * self.getUpper(flrg) )
				lo.append( self.getProbability(flrg) * mvs[count] * self.getLower(flrg) )
				count = count + 1
			
			# gerar o intervalo
			ret.append( [ sum(lo), sum(up) ] )
				
		return ret
