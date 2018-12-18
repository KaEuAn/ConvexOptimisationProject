import numpy as np
from ..core.exceptions import ConditionsError
import math

class FirstOrderOracle:
    '''
    This class implements oracle for
    the following function:

    f(x) = (1 + exp(-ax - b) / (cx + d))

    it allows to compute 
    * for given point x: f(x)
    * for gimen point x: f'(x)

    where 
    * a and c are n-dimensional vectors
    * b and d are scalars
    '''
    def __init__(self, a, b, c, d):
        try:
            assert(a.shape[0] == c.shape[0])
        except ... :
            raise(ConditionsError('wrong conditions'))
        self.dimension = a.shape[0]
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        pass
    
    def func(self, x): 
        '''
        The result of the function will be following:
        res = (1 + exp(-ax - b))/(cx + d)
        '''
        try:
            assert(x.shape[0] == self.dimension)
        except ... :
            raise(ConditionsError(
                'wrong point x'
            ))
        
        y = -self.a.dot(x) - self.b
        z = self.c.dot(x) + self.d
        if(z == 0):
            raise(ConditionsError(
                'function is not defined for this point'
            ))

        res = (1 + math.e ** y)/z
        return(res)
    
    def first_derivative(self, x):
        ''' 
        The result of the function:
        res = -a exp(-ax - b)/(cx + d) -
            - c (1 + exp(-ax - b))/(cx + d)^2
        '''
        try:
            assert(x.shape[0] == self.dimension)
        except ... :
            raise(ConditionsError(
                'wrong point x'
            ))
        
        y = -self.a.dot(x) - self.b
        z = self.c.dot(x) + self.d
        if(z == 0):
            raise(ConditionsError(
                'function is not defined for this point'
            ))
        deriv = - self.a * (math.e ** y)/z - self.c * (1 + math.e ** y)/(z ** 2)

        return(deriv)
    
    def get_dimension(self):
        return(self.dimension)
    
class SecondOrderOracle(FirstOrderOracle):
    '''
    This class implements oracle for
    the following function:

    f(x) = (1 + exp(-ax - b) / (cx + d))

    it allows to compute 
    * for given point x: f(x)
    * for given point x: f'(x)
    * for given point x: f''(x)

    where 
    * a and c are n-dimensional vectors
    * b and d are scalars
    '''
    def __init__(self, a, b, c, d):
        super().__init__(a, b, c, d)

        # addin marices for second derivative
        a_rs = a.reshape((1, a.shape[0]))
        c_rs = c.reshape((1, c.shape[0]))
        self.aTa = a_rs.T.dot(a_rs)
        self.cTc = c_rs.T.dot(c_rs)
        self.aTc = a_rs.T.dot(c_rs)
        self.cTa = c_rs.T.dot(a_rs)

    
    def second_derivative(self, x):
        '''
        Implementation of Hessian
        '''
        try:
            assert(x.shape[0] == self.dimension)
        except ... :
            raise(ConditionsError(
                'wrong point x'
            ))
        
        y = -self.a.dot(x) - self.b
        ey = math.e ** (y)
        z = self.c.dot(x) + self.d
        if(z == 0):
            raise(ConditionsError(
                'function is not defined for this point'
            ))
        
        res = self.aTa * ey / z + \
        self.aTc * ey / (z ** 2) + \
        self.cTa * ey / (z ** 2) + \
        2 * self.cTc * (1 + ey)/ (z ** 3)

        return res