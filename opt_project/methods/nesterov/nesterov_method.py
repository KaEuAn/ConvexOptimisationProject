from . import approx_path
from ..gradient_descent.step_sizes import ConstantStepSize as default_ss
from ...core.exceptions import InitialPositionError

import numpy as np
from inspect import signature
from copy import deepcopy
        
        

class nesterov_gradient_descent():

    def __init__(self, oracle, constraints):
        self.oracle = oracle
        self.costraints = constraints
        self.pos = oracle.dimension
        self.alpha = default_ss(0.01)
        self.y_pos = self.pos
        self.prev_pos = self.pos

    def set_step_size(self, alpha):
        self.alpha = alpha
    
    def set_init_position(self, x):
        if self.costraints.satisfy(x):
            self.pos = x
            self.y_pos = x
        else:
            raise(InitialPositionError('wrong init position'))
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def make_step(self, iter_num):
        nablaF = self.get_gradient(self.pos)
        alpha =  self.alpha(self.y_pos, None, nablaF)
        prev_position = self.pos
        self.pos = self.y_pos - alpha * nablaF
        self.y_pos = self.pos + (iter_num / (iter_num + 3)) * (self.pos - self.prev_pos)
        self.prev_pos = prev_position
        

    def make(self, stop_criteria):
        path = approx_path()
        iters_num = 1
        path.Append(self.pos)
        while not stop_criteria(self.pos, self.oracle):
            self.make_step(iters_num)
            iters_num += 1
            print(self.pos)
            self.pos = self.costraints.projection(self.pos)
            print('proj', self.pos)
            path.Append(self.pos)
            # print(self.func(self.pos))
        return(path)