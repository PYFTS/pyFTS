import numpy as np
from pyFTS import *

def differential(original):
    n = len(original)
    diff = [ original[t-1]-original[t] for t in np.arange(1,n) ]
    diff.insert(0,0)
    return np.array(diff)

def trimf(x,parameters):
	if(x < parameters[0]):
		return 0
	elif(x >= parameters[0] and x < parameters[1]):
		return (x-parameters[0])/(parameters[1]-parameters[0])
	elif(x >= parameters[1] and x <= parameters[2]):
		return (parameters[2]-x)/(parameters[2]-parameters[1])
	else: 
		return 0

def trapmf(x, parameters):
		if(x < parameters[0]):
			return 0
		elif(x >= parameters[0] and x < parameters[1]):
			return (x-parameters[0])/(parameters[1]-parameters[0])
		elif(x >= parameters[1] and x <= parameters[2]):
			return 1
		elif(x >= parameters[2] and x <= parameters[3]):
			return (parameters[3]-x)/(parameters[3]-parameters[2])
		else: 
			return 0

def gaussmf(x,parameters):
		return math.exp(-0.5*((x-parameters[0]) / parameters[1] )**2)


def bellmf(x,parameters):
		return 1 / (1 + abs((xx - parameters[2])/parameters[0])**(2*parameters[1]))


def sigmf(x,parameters):
		return 1 / (1 + math.exp(-parameters[0] * (x - parameters[1])))


class FuzzySet:
	def __init__(self,name,mf,parameters,centroid):
		self.name = name
		self.mf = mf
		self.parameters = parameters
		self.centroid = centroid
		self.lower = min(parameters)
		self.upper = max(parameters)
        
	def membership(self,x):
		return self.mf(x,self.parameters)
    
	def __str__(self):
		return self.name + ": " + str(self.mf) + "(" + str(self.parameters) + ")"
    
class FLR:
	def __init__(self,LHS,RHS):
		self.LHS = LHS
		self.RHS = RHS
	
	def __str__(self):
		return str(self.LHS) + " -> " + str(self.RHS)
    
def fuzzyInstance(inst, fuzzySets):
    mv = np.array([ fs.membership(inst) for fs in fuzzySets])
    return mv


def fuzzySeries(data,fuzzySets):
	fts = []
	for item in data:
		mv = fuzzyInstance(item,fuzzySets)
		fts.append(fuzzySets[  np.argwhere(mv == max(mv) )[0,0] ])
	return fts


def generateNonRecurrentFLRs(fuzzyData):
	flrs = {}
	for i in range(2,len(fuzzyData)):
		tmp = FLR(fuzzyData[i-1],fuzzyData[i])
		flrs[str(tmp)] = tmp
	ret = [value for key, value in flrs.items()]
	return ret

def generateRecurrentFLRs(fuzzyData):
	flrs = []
	for i in range(2,len(fuzzyData)):
		flrs[i-1] = FLR(fuzzyData[i-1],fuzzyData[i])
	return flrs
