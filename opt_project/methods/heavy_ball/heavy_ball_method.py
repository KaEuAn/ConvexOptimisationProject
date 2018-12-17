from . import gd_superclass
from . import approx_path
import numpy as np

class InitialPositionError(Exception):
    pass

class heavy_ball_method(gd_superclass):

    def __init__(self, oracle, constraints, beta):
        super().__init__(oracle.func, oracle.dimension)
        self.oracle = oracle
        self.costraints = constraints
        self.previous_val = self.pos

    def set_alpha(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def get_alpha(self):
        return (self.alpha, self.beta)

    def make_step(self):
	nablaF = self.get_gradient(self.pos)
        prev_val = self.pos
        self.pos = self.pos - self.get_alpha() * nablaF - self.beta * (self.pos - self.previous_val)
        self.previous_val = prev_val
                

    def make(self, stop_criteria):
        path = approx_path()
        path.Append(self.pos)
        while not stop_criteria():
            self.make_step()
            self.pos = self.costraints.projection(self.pos)
            path.Append(self.pos)
            # print(self.func(self.pos))
        return(path)
