class FLR:
    def __init__(self, LHS, RHS):
        self.LHS = LHS
        self.RHS = RHS

    def __str__(self):
        return str(self.LHS) + " -> " + str(self.RHS)

def generateNonRecurrentFLRs(fuzzyData):
    flrs = {}
    for i in range(2,len(fuzzyData)):
        tmp = FLR(fuzzyData[i-1],fuzzyData[i])
        flrs[str(tmp)] = tmp
    ret = [value for key, value in flrs.items()]
    return ret

def generateRecurrentFLRs(fuzzyData):
    flrs = []
    for i in np.arange(1,len(fuzzyData)):
        flrs.append(FLR(fuzzyData[i-1],fuzzyData[i]))
    return flrs
