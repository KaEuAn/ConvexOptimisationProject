import numpy as np
from copy import copy

import abc

class StepSize:
    '''
    This class makes abstract interface of 
    Gradient Descent Step Size

    The __call__ method must take
    x_0, f(x_0), f'(x_0), dir, f, grad_f
    where h is the direction of 
    the next point finding
    (as usual, h=-f'(x_0))
    f is the original function
    grad_f is the original differential of the function
    '''
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def __call__(self, x, val, grad_val, h):
        pass

class ConstantStepSize(StepSize) :
    '''
    This class implements the realisation of 
    Constant step size rule for selecting step size
    alpha is just constant
    '''
    def __init__(self, alpha=0.01) :
        super().__init__()
        self.alpha = alpha
    
    def __call__(self, x=None, val=None, grad_val=None, h=None):
        return self.alpha
    
class ArmijoStepSize(StepSize) :
    '''
    This class implements the realisation of 
    Armiho rule for selecting step size

    *Note*, that this method can be used only for
    unconstrainted optimization

    This rule includes the requirement of 
    sufficient decrease
    
    This means that the new approach point 
    x_{k + 1} = x_{k} + h * alpha
    satisfies the following conditions:
    1. f_{x_{k + 1}} <= f_{x_k} + beta * alpha * (f_{x_k}^' * h)

    The beta parameter are defined by users
    To select the appropriate alpha, we use
    initial assumption about alpha meaning (init_alpha)
    and then try to chose the alpha
    with step-by-step decrease of init_alpha
    with dec_c coefficient
    '''
    def __init__(self, oracle, constraints, dec_c=0.9, init_alpha=0.1, beta=0.0001, max_it = 20) :
        super().__init__()
        self.oracle = oracle
        self.constraints = constraints
        self.dec_c = dec_c
        self.init_alpha = init_alpha
        self.beta = beta
        self.max_it = max_it
    
    def __call__(self, x, val, grad_val, h=None):
        max_it = copy(self.max_it)
        f = self.oracle.func
        if val == None:
            val = f(x)
        alpha = self.init_alpha

        # Check violating sufficient decrease and curvature conditions 
        x_new = 0
        while True:
            if max_it < 0 :
                break
            if not self.constraints.satisfy(x - grad_val * alpha):
                x_new = self.constraints.projection(x - grad_val, save_state=True)
            else:
                x_new = x - alpha * grad_val
            if f(x_new) <= f(x) + alpha * self.beta * grad_val.dot(x_new - x):
                break
            
            alpha *= self.dec_c
            max_it -= 1
        print(x_new)
        return alpha
        
