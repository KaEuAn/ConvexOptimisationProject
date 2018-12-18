import numpy as np
from ...core.exceptions import GessianMatrixReversibilityError

class newton_stop_criteria:
    def __init__(self, oracle, eps = 0.001):
        self.oracle = oracle
        self.eps = eps
    
    def __call__(self, x, val, grad_val, hess_val):
        if grad_val == None:
            grad_val = self.oracle.first_derivative(x)
        if hess_val == None:
            hess_val = self.oracle.second_derivative(x)
        try:
            hess_inv = np.linalg.inv(hess_val)
        except ... :
            raise(GessianMatrixReversibilityError(
                'Matrix is not reversible'
            ))
        square_lambda = grad_val.dot(hess_inv.dot(grad_val))
        return(square_lambda/2 < self.eps)

class constrained_newton_stop_criteria:
    def __init__(self, oracle, eps = 0.001):
        self.oracle = oracle
        self.eps = eps
    
    def __call__(self, x, val, grad_val, hess_val, h):
        if hess_val == None:
            hess_val = self.oracle.second_derivative(x)
        square_lambda = h.dot(hess_val.dot(h))
        return(square_lambda/2 < self.eps)
        
