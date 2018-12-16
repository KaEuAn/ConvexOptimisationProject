from . import approx_path
from .step_sizes import ConstantStepSize as default_ss

import numpy as np
from inspect import signature
from copy import deepcopy

class InitialPositionError(Exception):
    pass
        
        

class simple_gradient_descent():

    class param_detector(object):
        def __init__(self, func):
            sig = signature(func)
            if 'value' in str(sig) :
                self.val = True
            else :
                self.val = False
            
            if 'func_value' in str(sig):
                self.fval = True
            else :
                self.fval = False
            
            self.func = func
        
        def __call__(self, suppl) :
            if self.val and self.fval:
                return(self.func(suppl.pos, suppl.oracle.func(suppl.pos)))
            
            elif self.val :
                return(self.func(suppl.pos))
            
            else :
                return(self.func())

    def __init__(self, oracle, constraints):
        self.oracle = oracle
        self.costraints = constraints
        self.pos = oracle.dimension
        self.alpha = default_ss(0.01)

    def set_step_size(self, alpha):
        self.alpha = alpha
    
    def set_init_position(self, x):
        if self.costraints.satisfy(x):
            self.pos = x
        else:
            raise(InitialPositionError('wrong init position'))
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def make_step(self):
        nablaF = self.get_gradient(self.pos)
        alpha =  self.alpha(self.pos, None, nablaF)
        self.pos = self.pos - alpha * nablaF

    def make(self, stop_criteria):
        stop_criteria = self.param_detector(stop_criteria)
        path = approx_path()
        path.Append(self.pos)
        while not stop_criteria(self):
            self.make_step()
            self.pos = self.costraints.projection(self.pos)
            path.Append(self.pos)
            # print(self.func(self.pos))
        return(path)
        