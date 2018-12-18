from . import approx_path
from ..gradient_descent.step_sizes import ConstantStepSize as default_ss
from ...core.exceptions import InitialPositionError

import numpy as np
from inspect import signature
from copy import deepcopy
        
        

class bfgs_descent():

    def __init__(self, oracle, constraints):
        self.oracle = oracle
        self.costraints = constraints
        self.alpha = default_ss(0.01)

    def set_init_position(self, x):
        if self.costraints.satisfy(x):
            self.pos = x
            self.grad = self.oracle.first_derivative(self.pos)
            self.B = np.eye(len(self.pos))
            self.prev_pos = self.pos - np.ones(len(self.pos)) * 0.5
            self.prev_grad = self.oracle.first_derivative(self.pos) - np.ones(len(self.pos)) * 0.3
        else:
            raise(InitialPositionError('wrong init position'))
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def make_step(self):
        s = self.pos - self.prev_pos
        self.prev_pos = self.pos
        new_grad = self.oracle.first_derivative(self.pos)
        y = new_grad - self.prev_grad
        self.pos = self.pos - np.linalg.inv(self.B).dot(self.prev_grad)
        self.B = self.B - (self.B.dot(s.dot(np.transpose(s).dot(self.B)))) / (np.transpose(s).dot(self.B.dot(s))) + y.dot(np.transpose(y))/(np.transpose(y).dot(s))
        

    def make(self, stop_criteria):
        path = approx_path()
        path.Append(self.pos)
        while not stop_criteria(self.pos, self.oracle):
            self.make_step()
            print(self.pos)
            print(self.B)
            self.pos = self.costraints.projection(self.pos)
            path.Append(self.pos)
            # print(self.func(self.pos))
        return(path)