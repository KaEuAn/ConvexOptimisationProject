from . import gd_superclass
import numpy as np

class InitialPositionError(Exception):
    pass

class simple_gradient_descent(gd_superclass):

    def __init__(self, oracle, constraints):
        super().__init__(oracle.func, oracle.dimension)
        self.oracle = oracle
        self.costraints = constraints

    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def get_alpha(self):
        return self.alpha

    def make(self, stop_criteria):
        while not stop_criteria():
            self.make_step()
            self.pos = self.costraints.projection(self.pos)
            # print(self.func(self.pos))
        return(self.pos)
        