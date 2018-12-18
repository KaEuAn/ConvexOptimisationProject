from . import approx_path
import numpy as np
from ...core.exceptions import InitialPositionError

class heavy_ball_method:

    def __init__(self, oracle, constraints):
        self.oracle = oracle
        self.costraints = constraints

    def set_alpha(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def get_alpha(self):
        return (self.alpha, self.beta)
    
    def set_init_position(self, x):
        if self.costraints.satisfy(x):
            self.pos = x
            self.previous_val = self.pos
        else:
            raise(InitialPositionError('wrong init position'))

    def make_step(self):
        nablaF = self.get_gradient(self.pos)
        prev_val = self.pos
        self.pos = self.pos - self.alpha * nablaF - self.beta * (self.pos - self.previous_val)
        self.previous_val = prev_val
                

    def make(self, stop_criteria):
        path = approx_path()
        path.Append(self.pos)
        while not stop_criteria(self.pos, self.oracle):
            self.make_step()
            self.pos = self.costraints.projection(self.pos)
            path.Append(self.pos)
            # print(self.func(self.pos))
        return(path)
