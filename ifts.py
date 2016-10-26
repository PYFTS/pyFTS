import numpy as np
from pyFTS import *

class IntervalFTS(hofts.HighOrderFTS):
	def __init__(self,name):
		super(IntervalFTS, self).__init__("IFTS")
		self.name = "Interval FTS"
		self.detail = "Silva, P.; Guimarães, F.; Sadaei, H."
		self.flrgs = {}
		self.isInterval = True
    
	def getUpper(self,flrg):
		if flrg.strLHS() in self.flrgs:
			tmp = self.flrgs[ flrg.strLHS() ]
			ret = max(np.array([self.setsDict[s].upper for s in tmp.RHS]))
		else:
			ret = flrg.LHS[-1].upper
		return ret
		
	def getLower(self,flrg):
		if flrg.strLHS() in self.flrgs:
			tmp = self.flrgs[ flrg.strLHS() ]
			ret = min(np.array([self.setsDict[s].lower for s in tmp.RHS]))
		else:
			ret = flrg.LHS[-1].lower
		return ret
		
	def getSequenceMembership(self, data, fuzzySets):
		mb = [ fuzzySets[k].membership( data[k] )  for k in np.arange(0,len(data))  ]
		return mb
		
	def buildTree(self,node, lags, level):
		if level >= self.order:
			return
			
		for s in lags[level]:				
			node.appendChild(tree.FLRGTreeNode(s))
			
		for child in node.getChildren():
			self.buildTree(child,lags,level+1)
    	
	def forecast(self,data):
		
		ndata = np.array(data)
		
		l = len(ndata)
		
		ret = []
		
		for k in np.arange(self.order,l):
			
			flrs = []
			mvs = []
			
			up = []
			lo = []
			
			# Achar os conjuntos que tem pert > 0 para cada lag
			count = 0
			lags = {}
			if self.order > 1:
				subset = ndata[k-self.order : k ]
				
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
			norm = sum(mvs)
			ret.append( [ sum(lo)/norm, sum(up)/norm ] )
				
		return ret
