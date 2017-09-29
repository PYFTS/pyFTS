import numpy as np
from pyFTS.common import Membership
from pyFTS.nonstationary import common,pertubation,util
import importlib

importlib.reload(util)

uod = np.arange(0,20,0.1)

kwargs = {'location': pertubation.linear, 'location_params': [1,0],
          'width': pertubation.linear, 'width_params': [1,0]}

mf1 = common.MembershipFunction('A1',Membership.trimf,[0,1,2], **kwargs)
mf2 = common.MembershipFunction('A2',Membership.trimf,[1,2,3], **kwargs)
mf3 = common.MembershipFunction('A3',Membership.trimf,[2,3,4], **kwargs)
mf4 = common.MembershipFunction('A4',Membership.trimf,[3,4,5], **kwargs)

sets = [mf1, mf2, mf3, mf4]

util.plot_sets(uod, sets)
