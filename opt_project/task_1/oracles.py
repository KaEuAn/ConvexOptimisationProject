import numpy as np
import math

class FirstOrderOracle:
    def __init__(self, a, b):
        self.dimension = a.shape[0]
        self.a = a
        self.b = b
    
    def func(self, x):
        assert(x.shape[0] == self.a.shape[0])
        #res = math.e ** (a @ x + b) / (1 + math.e ** (a @ x + b))
        res = - (self.a @ x + self.b)
        return(res)
    
    def first_derivative(self, x):
        assert(x.shape[0] == self.a.shape[0])
        '''
        func_res = self.func(x)
        res = self.a * func_res
        '''
        return(-self.a)
    
    def get_dimension(self):
        return(self.dimension)

class SecondOrderOracle(FirstOrderOracle):

    def second_derivative(self, x):
        #return(np.outer(self.first_derivative, self.a))
        return np.zeros((x.shape[0], x.shape[0]))