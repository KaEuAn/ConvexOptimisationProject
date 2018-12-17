from . import NoequalLinearConstraints
import numpy as np
from copy import deepcopy
from ..core.exceptions import ConditionsError

class TaskTwoLinearConstraints(NoequalLinearConstraints):
    '''
    This class specify linear constraints,
    definded as follows:
    F * x <= g
    c * x + d > 0

    Here F is n*m - dim matrix
    x is m - dim unknown vector
    c is m - dim vector
    d is constant
    '''
    def __init__(self, F, g, c, d):
        '''
        here we transform initial data 
        as follows:
        * F -> F' = (r1, r2 , .. rn , -c)^T
        where ri are rows of F'
        * g -> g' = (g1, g2, .. gn, d)
        
        So, we get the following 
        UnequallyLinearConstrains conditions:
        F' * x < g'
        '''
        if F.shape[1] != c.shape[0]:
            raise(ConditionsError('wrong conditions')) 
        c_rs = c.reshape((1, c.shape[0]))
        F_res = np.append(F, -c_rs, axis=0)
        g_res = np.append(g, [d,], axis=0)
        super().__init__(F_res, g_res)

        