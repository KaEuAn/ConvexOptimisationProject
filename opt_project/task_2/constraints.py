from . import LinearConstraints
import numpy as np

class InitialisationError(Exception):
    pass

class TaskTwoLinearConstraints(LinearConstraints):
    '''
    This class specify linear constraints,
    definded as follows:
    F * x <= g
    a * x + b <= y
    c * x + d <= z
    c * x + d > 0

    Here F is n*m - dim matrix
    x is m - dim unknown vector
    a is m - dim vector
    b is constant
    c is m - dim vector
    d is constant

    y and z are values we want to restrict
    '''
    def __init__(self, F, g, a, b, c, d):
        '''
        here we transform initial data 
        as follows:
        * F -> F' = (f1, f2, .., fm , 0, 0),
        where fi are columns of initial F matrix
        * e = (0, ... 0, 1)
        * F' -> F'' = (r1, r2 , .. rn , e)^T
        where ci are rows of F'
        * a' = (a1, ..., am, -1, 0)
        * c' = (c1, ..., cm, 0, -1)
        * b'' = (-b, -d)
        * A' = (a', c')^T
        * g' = (g1, ..., gn, 0)
        * x' = (x1, ..., xm, y, z)
        So, with reduce the description of constraints 
        to following:
        F'' * x' < g'
        A' * x = b''

        '''
        zero_columns = np.zeros((F.shape[0], 2))
        F = np.append(F, zero_columns, axis=1)
        e = np.zeros(F.shape[1])
        e[e.shape[0] - 1] = -1
        e = e.reshape((1, e.shape[0]))
        F_res = np.append(F, e, axis=0)
        a = np.append(a, [-1, 0]).reshape((1, a.shape[0] + 2))
        c = np.append(c, [0, -1]).reshape((1, c.shape[0] + 2))
        A_res = np.append(a, c, axis=0)
        b_res = np.array([-b, -d])
        g_res = np.append(g, [0,], axis=0)
        super().__init__(F_res, g_res, A_res, b_res)
    
    def projection(self, x):
        '''
        Here we make the projection
        of x with respect to 
        self.point vector
        '''
        assert(x.shape[0] == 2)
        try:
            self.point[-2:] = x
        except AttributeError:
            raise InitialisationError(
                "The initial point of the set has't been initialised")
        self.point = super().projection(self.point)
        return self.point[-2:]
    
    def initialise(self, x=None):
        '''
        Initialisation of self.point vector
        if x == None, then self.point
        will be initialized with 
        projection of zero vector
        on the constrainted set
        '''
        if x == None:
            x = np.zeros(self.F.shape[1])
        if self.satisfy(x):
            self.point = x
        self.point = super().projection(x)
        return self.point[-2:]


        