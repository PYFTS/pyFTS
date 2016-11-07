import numpy as np
import pandas as pd
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
		
		#print(ndata)
		
		l = len(ndata)
		
		ret = []
		
		for k in np.arange(self.order-1,l):
			
			affected_flrgs = []
			affected_flrgs_memberships = []
			norms = []
			
			up = []
			lo = []
			
			# Find the sets which membership > 0 for each lag
			count = 0
			lags = {}
			if self.order > 1:
				subset = ndata[k-(self.order-1) : k+1 ]
				
				for instance in subset:
					mb = common.fuzzyInstance(instance, self.sets)
					tmp = np.argwhere( mb )
					idx = np.ravel(tmp) #flatten the array
					
					if idx.size == 0:	# the element is out of the bounds of the Universe of Discourse
						if instance <= self.sets[0].lower:
							idx = [0]
						if instance >= self.sets[-1].upper:
							idx = [len(self.sets)-1]
						
					lags[count] = idx 
					count = count + 1
					
					
				# Build the tree with all possible paths
				
				root = tree.FLRGTreeNode(None)
				
				self.buildTree(root,lags,0)
				
				# Trace the possible paths and build the PFLRG's
				
				for p in root.paths():
					path = list(reversed(list(filter(None.__ne__, p))))
					flrg = hofts.HighOrderFLRG(self.order)
					for kk in path: flrg.appendLHS(self.sets[ kk ])
					
					##
					affected_flrgs.append( flrg )
					
					# Find the general membership of FLRG
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
			if norm == 0:
				ret.append( [ 0, 0 ] )
			else:
				ret.append( [ sum(lo)/norm, sum(up)/norm ] )
				
		return ret
		
	def forecastAhead(self,data,steps):
		ret = [[data[k],data[k]] for k in np.arange(len(data)-self.order,len(data))]
		for k in np.arange(self.order,steps):
			if ret[-1][0] <= self.sets[0].lower and ret[-1][1] >= self.sets[-1].upper:
				ret.append(ret[-1])
			else:
				lower = self.forecast( [ret[x][0] for x in np.arange(k-self.order,k)] )
				upper = self.forecast( [ret[x][1] for x in np.arange(k-self.order,k)] )
				ret.append([np.min(lower),np.max(upper)])
			
		return ret
		
	def getGridClean(self,resolution):
		grid = {}
		for sbin in np.arange(self.sets[0].lower,self.sets[-1].upper,resolution):
			grid[sbin] = 0
			
		return grid
		
	def gridCount(self, grid, resolution, interval):
		for sbin in sorted(grid):
			if sbin >= interval[0] and (sbin + resolution) <= interval[1]:
				grid[sbin] = grid[sbin] + 1
		return grid
		
	def forecastDistributionAhead(self,data,steps,resolution):
		
		ret = []
		
		intervals = self.forecastAhead(data,steps)
		
		for k in np.arange(self.order,steps):
			
			grid = self.getGridClean(resolution)
			grid = self.gridCount(grid,resolution, intervals[k])
			
			for qt in np.arange(1,50,2):
				#print(qt)
				qtle_lower = self.forecast([intervals[x][0] + qt*(intervals[x][1]-intervals[x][0])/100 for x in np.arange(k-self.order,k)] )
				grid = self.gridCount(grid,resolution, np.ravel(qtle_lower))
				qtle_upper = self.forecast([intervals[x][1] - qt*(intervals[x][1]-intervals[x][0])/100 for x in np.arange(k-self.order,k)] )
				grid = self.gridCount(grid,resolution, np.ravel(qtle_upper))
			qtle_mid = self.forecast([intervals[x][0] + (intervals[x][1]-intervals[x][0])/2 for x in np.arange(k-self.order,k)] )
			grid = self.gridCount(grid,resolution, np.ravel(qtle_mid))
			
			tmp = np.array([ grid[k] for k in sorted(grid) ])
			
			ret.append( tmp/sum(tmp) )
			
		grid = self.getGridClean(resolution)
		df = pd.DataFrame(ret, columns=sorted(grid))
		return df
		
	def __str__(self):
		tmp = self.name + ":\n"
		for r in sorted(self.flrgs):
			p = round(self.flrgs[r].frequencyCount / self.globalFrequency,3)
			tmp = tmp + "(" + str(p) + ") " + str(self.flrgs[r]) + "\n"
		return tmp
