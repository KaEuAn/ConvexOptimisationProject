from . import approx_path
from .step_sizes import ConstantStepSize as default_ss
from ...core.exceptions import InitialPositionError

import numpy as np
from inspect import signature
from copy import deepcopy
        
        

class bfgs_descent():

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
        self.grad = self.oracle.first_derevative(self.pos)
        B = np.eye(len(self.pos))

    def set_init_position(self, x):
        if self.costraints.satisfy(x):
            self.pos = x
        else:
            raise(InitialPositionError('wrong init position'))
    
    def get_gradient(self, a):
        return self.oracle.first_derivative(a)
    
    def make_step(self):
        nablaF = self.get_gradient(self.pos)
        prev_position = selp.pos
        self.pos = self.pos - np.linalg.inv(B).dot(grad)
        new_grad = self.oracle.first_derevative(self.pos)
        y = new_grad - self.grad
        s = self.pos - prev_position
        B = B - (B.dot(s.dot(np.transpose(s).dot(B)))) / (np.transpose(s).dot(B.dot(s))) + y.dot(np.transpose(y))/(np.transpose(y).dot(s))
        

    def make(self, stop_criteria):
        stop_criteria = self.param_detector(stop_criteria)
        path = approx_path()
        iters_num = 1
        path.Append(self.pos)
        while not stop_criteria(self):
            self.make_step()
            self.pos = self.costraints.projection(self.pos)
            path.Append(self.pos)
            # print(self.func(self.pos))
        return(path)