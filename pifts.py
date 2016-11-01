import numpy as np
from pyFTS import *

class ProbabilisticFLRG(hofts.HighOrderFLRG):
	def __init__(self,order):
		super(ProbabilisticFLRG, self).__init__(order)
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
				tmp2 = tmp2 + ", "
			tmp2 = tmp2 + c + "(" + str(round(self.RHS[c]/self.frequencyCount,3)) + ")"
		return self.strLHS() + " -> " + tmp2

class ProbabilisticIntervalFTS(ifts.IntervalFTS):
	def __init__(self,name):
		super(ProbabilisticIntervalFTS, self).__init__("PIFTS")
		self.shortname = "PIFTS " + name
		self.name = "Probabilistic Interval FTS"
		self.detail = "Silva, P.; Guimarães, F.; Sadaei, H."
		self.flrgs = {}
		self.globalFrequency = 0
		self.isInterval = True
		
	def generateFLRG(self, flrs):
		flrgs = {}
		l = len(flrs)
		for k in np.arange(self.order +1, l):
			flrg = ProbabilisticFLRG(self.order)
			
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
		if flrg.strLHS() in self.flrgs:
			return self.flrgs[ flrg.strLHS() ].frequencyCount / self.globalFrequency
		else:
			return 1/ self.globalFrequency
		
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
		
		for k in np.arange(self.order-1,l):
			
			affected_flrgs = []
			affected_flrgs_memberships = []
			norms = []
			
			up = []
			lo = []
			
			# Achar os conjuntos que tem pert > 0 para cada lag
			count = 0
			lags = {}
			if self.order > 1:
				subset = ndata[k-(self.order-1) : k+1 ]
				for instance in subset:
					mb = common.fuzzyInstance(instance, self.sets)
					tmp = np.argwhere( mb )
					idx = np.ravel(tmp) #flat the array
					lags[count] = idx 
					count = count + 1				
					
				# Constrói uma árvore com todos os caminhos possíveis
				
				root = tree.FLRGTreeNode(None)
				
				self.buildTree(root,lags,0)
				
				# Traça os possíveis caminhos e costrói as PFLRG's
				
				for p in root.paths():
					path = list(reversed(list(filter(None.__ne__, p))))
					flrg = hofts.HighOrderFLRG(self.order)
					for kk in path: flrg.appendLHS(self.sets[ kk ])
					
					##
					affected_flrgs.append( flrg )
					
					# Acha a pertinência geral de cada FLRG
					affected_flrgs_memberships.append(min(self.getSequenceMembership(subset, flrg.LHS)))
			else:
				
				mv = common.fuzzyInstance(ndata[k],self.sets) # get all membership values
				tmp = np.argwhere( mv ) # get the indices of values > 0
				idx = np.ravel(tmp) # flatten the array
				for kk in idx:
					flrg = hofts.HighOrderFLRG(self.order)
					flrg.appendLHS(self.sets[ kk ])
					affected_flrgs.append( flrg  )
					affected_flrgs_memberships.append(mv[kk])
			
			count = 0
			for flrg in affected_flrgs:
				# achar o os bounds de cada FLRG, ponderados pela probabilidade e pertinência
				norm = self.getProbability(flrg) * affected_flrgs_memberships[count]
				up.append( norm * self.getUpper(flrg) )
				lo.append( norm * self.getLower(flrg) )
				norms.append(norm)
				count = count + 1
			
			# gerar o intervalo
			norm = sum(norms)
			ret.append( [ sum(lo)/norm, sum(up)/norm ] )
				
		return ret
	
	def forecastAhead(self,data,steps):
		ret = [[data[k],data[k]] for k in np.arange(len(data)-self.order,len(data))]
		for k in np.arange(self.order,steps):
			lower = self.forecast( [ret[x][0] for x in np.arange(k-self.order,k)] )
			upper = self.forecast( [ret[x][1] for x in np.arange(k-self.order,k)] )
			ret.append([np.min(lower),np.max(upper)])
			
		return ret
			
	def __str__(self):
		tmp = self.name + ":\n"
		for r in sorted(self.flrgs):
			p = round(self.flrgs[r].frequencyCount / self.globalFrequency,3)
			tmp = tmp + "(" + str(p) + ") " + str(self.flrgs[r]) + "\n"
		return tmp
