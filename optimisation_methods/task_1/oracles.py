import numpy as np
import math

class FirstOrderOracle:
    def __init__(self, a, b):
        self.dimension = a.shape[0]
        self.a = a
        self.b = b
    
    def func(self, x):
        assert(x.shape[0] == a.shape[0])
        a = self.a
        b = self.b
        res = math.e ** (a @ x + b) / (1 + math.e ** (a @ x + b))
        return(res)
    
    def first_derivative(self, x):
        assert(x.shape[0] == self.a.shape[0])
        func_res = self.func(x)
        res = self.a * func_res
        return(res)

class SecondOrderOracle(FirstOrderOracle):

    def second_derivative(self, x):
        return(np.outer(self.first_derivative, self.a))