import numpy as np
from pyFTS import *

class ProbabilisticIntervalFLRG:
	def __init__(self,order):
		self.LHS = []
		self.RHS = []
		self.RHSfreqs = {}
		self.order = order
		self.frequencyCount = 0
		self.strlhs = ""

	def appendRHS(self,c):
		self.RHS.append(c)
		self.frequencyCount = self.frequencyCount + 1
		if c.name in self.RHSfreqs:
			self.RHSfreqs[c.name] = self.RHSfreqs[c.name] + 1
		else:
			self.RHSfreqs[c.name] = 1
		
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

class ProbabilisticIntervalFTS(ifts.IntervalFTS):
	def __init__(self,name):
		super(IntervalFTS, self).__init__(1,name)
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
					
					flrs.append(flrg)
					
					# Acha a pertinência geral de cada FLRG
					mvs.append(min(self.getSequenceMembership(subset, flrg.LHS)))
			else:
				
				mv = common.fuzzyInstance(ndata[k],self.sets)
				tmp = np.argwhere( mv )
				idx = np.ravel(tmp)
				for kk in idx:
					flrg = hofts.HighOrderFLRG(self.order)
					flrg.appendLHS(self.sets[ kk ])
					flrs.append(flrg)
					mvs.append(mv[kk])
			
			count = 0
			for flrg in flrs:
				# achar o os bounds de cada FLRG, ponderados pela pertinência
				up.append( mvs[count] * self.getUpper(flrg) )
				lo.append( mvs[count] * self.getLower(flrg) )
				count = count + 1
			
			# gerar o intervalo
			ret.append( [ sum(lo), sum(up) ] )
				
		return ret
