import numpy as np
from ...core.exceptions import InitialPositionError
from ...core.exceptions import GessianMatrixReversibilityError
from ...core.exceptions import ConditionsError
from . import approx_path
from copy import deepcopy
import sympy 

class NewtonMethod:

    eps_0 = 1e-1
    grow_factor = 1.2

    def _make_newton_step(self):
        A = self.oracle.second_derivative(self.pos)
        try:
            A_inv = np.linalg.inv(A)
        except :
            eps = self.eps_0
            while True:
                A = A + np.eye(A.shape[0])*eps
                try:
                    A_inv = np.linalg.inv(A)
                except :
                    eps = self.grow_factor * eps
                    continue
                break
        B = self.oracle.first_derivative(self.pos)
        h = -A_inv.dot(B)
        return(h)

    def set_init_position(self, x):
        self.pos = x

    def __init__(self, oracle, constraints):
        self.oracle = oracle
        self.constraints = constraints
    
    def set_step_size(self, alpha):
        self.alpha = alpha
    
    def make(self, stop_criteria):
        path = approx_path()
        path.Append(self.pos)
        while True:
            h = self._make_newton_step()
            if stop_criteria(self.pos, self.oracle):
                break
            nablaF = self.oracle.first_derivative(self.pos)
            alpha = self.alpha(self.pos, None, nablaF)   
            self.pos = self.pos + alpha * h
            # print(self.pos)
            self.pos = self.constraints.projection(self.pos)
            path.Append(self.pos)
        path.Append(self.pos)
        return(path)


        