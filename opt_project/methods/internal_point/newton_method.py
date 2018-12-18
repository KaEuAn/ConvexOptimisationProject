import numpy as np
from ...core.exceptions import InitialPositionError
from ...core.exceptions import GessianMatrixReversibilityError
from ...core.exceptions import ConditionsError
from . import approx_path
from copy import deepcopy
import sympy 

class UnconstrainedNewtonMethod:
    '''
    This class implements Newton Method
    for the solution unconstrained optimisation
    problem:

    minimize f(x)
     x \in R^n
    
    We assume, that f: R^n -> R is convex and 
    twice continiously differentiable 
    '''
    def _make_newton_step(self):
        A = self.oracle.second_derivative(self.pos)
        try:
            A_inv = np.linalg.inv(A)
        except ... :
            raise(GessianMatrixReversibilityError(
                'Matrix is not reversible'
            ))
        B = self.oracle.first_derivative(self.pos)
        h = -A_inv.dot(B)
        return(h)

    def set_init_position(self, x):
        self.pos = x

    def __init__(self, oracle):
        self.oracle = oracle
    
    def set_step_size(self, alpha):
        self.alpha = alpha
    
    def make(self, stop_criteria):
        path = approx_path()
        path.Append(self.pos)
        while True:
            h = self._make_newton_step()
            if stop_criteria(deepcopy(self.pos), None, None, None):
                break
            alpha = self.alpha(deepcopy(self.pos), None, None, h)
            self.pos = self.pos + alpha * h
            path.Append(self.pos)
        path.Append(self.pos)
        return(path)

class ConstrainedNewtonMethod:
    '''
    This class implements NewtonMethod
    for the solution equality constrained 
    optimization problem:

    minimize f(x)
    s.t. Ax = b

    We assume, that 
    * f: R^n -> R is convex and 
    twice continiously differentiable
    * A \in M_{p, n}(R)

    Also we assume, that the task is feasible, 
    this means, that optimal solution x* exists
    '''
    def _make_newton_step(self):
        # here we compute hessian of the function
        hess = self.oracle.second_derivative(self.pos)
        grad = self.oracle.first_derivative(self.pos)

        # here we create following matrix
        '''
        Hess | A^T
        _____|____
        A    | 0
             '
        '''
        M = np.zeros(
            (hess.shape[0] + self.A.shape[0], 
            hess.shape[0] + self.A.shape[0]))
        M[:hess.shape[0], :hess.shape[0]] = hess
        M[hess.shape[0], :hess.shape[0]] = self.A
        M[:hess.shape[0], hess.shape[0]:] = self.A.T

        # here we create the following vector:
        '''
        u = [-f'(self.pos), 0, ...]
        '''
        u = np.zeros(hess.shape[0] + self.A.shape[0])
        u[:hess.shape[0]] = -grad

        try:
            M_inv = np.linalg.inv(M)
        except ... :
            raise(GessianMatrixReversibilityError(
                'KKT Matrix is not reversible'
            ))
        res = M_inv.dot(u)
        return(res[:hess.shape[0]])

    def __init__(self, oracle, A, b):

        # check for the linear system A x = b is compatible
        assert(A.shape[0] == b.shape[0])

        A_rank = np.linalg.matrix_rank(A)
        A_augmented = np.append(A, b.reshape(b.shape[0], 1), axis=1)
        A_augmented_rank = np.linalg.matrix_rank(A_augmented)

        # Kronecker–Capelli theorem conditions:
        if A_rank == A_augmented_rank :
            # the system has solutions
            pass
        else :
            # the system has no solutions
            raise ConditionsError(
                'The conditions are not feasible')
        
        # reducing A matrix to independent rows
        _, indep_rows_numbers = sympy.Matrix(A).T.rref()
        indep_rows_numbers = list(indep_rows_numbers)
        A_indep = A[indep_rows_numbers]
        b_indep = b[indep_rows_numbers]

        self.A = A_indep
        self.b = b_indep
        self.oracle = oracle
    
    def set_init_position(self, x):
        try:
            assert(self.A.dot(x) == self.b)
        except:
            raise(InitialPositionError(
                'The initial point is wrong'
            ))
        self.pos = x
    
    def set_step_size(self, alpha):
        self.alpha = alpha
    
    def make(self, stop_criteria):
        path = approx_path()
        path.Append(self.pos)
        while True:
            h = self._make_newton_step()
            if stop_criteria(deepcopy(self.pos), None, None, None, h):
                break
            alpha = self.alpha(deepcopy(self.pos), None, None, h)
            self.pos = self.pos + alpha * h
            path.Append(self.pos)
        path.Append(self.pos)
        return(path)
    


        