import numpy as np
import math

class FirstOrderOracle:
    def __init__(self):
        self.dimension = 2
        pass
    
    def func(self, x): 
        '''
        let's assume , that x = np.array([y, z]),
        so the result of the function will be following:
        res = (1 + e^y)/(z * e^y)
        '''
        assert(x.shape[0] == 2)
        assert(x[1] != 0)
        res = (1 + math.e ** x[0])/(x[1] * math.e ** x[0])
        return(res)
    
    def first_derivative(self, x):
        '''
        let's assume, that x = (y, z), 
        here is the result of this function:
        res = (
            - e^(-y) / z,
            - (1 + e^(-y))/ z^2
        )
        '''
        assert(x.shape[0] == 2)
        assert(x[1] != 0)
        deriv = np.array([
            - (math.e ** (- x[0]))/x[1],
            - (1 + math.e **(- x[0]))/(x[1] ** 2)
        ])
        return(deriv)
    
    def get_dimension(self):
        return(self.dimension)
    
class SecondOrderOracle(FirstOrderOracle):
    
    def second_derivative(self, x):
        '''
        let's assume, that x = (y, z), 
        here is the result of this function:
        res = ( (e^(-y) / z,   e^(-y)/ z^2),
                (e^(-y)/ x^2,  (1 + e^(-y))/ z^3))
        '''
        assert(x.shape[0] == 2)
        assert(x[1] != 0)
        sderiv = np.array([
            [
                (math.e ** (-x[0]))/x[1], 
                (math.e ** (-x[0]))/(x[1] ** 2)], 
            [
                (math.e ** (-x[0]))/(x[1] ** 2), 
                (1 + math.e **(- x[0]))/(x[1] ** 3)]])
        return(sderiv)

