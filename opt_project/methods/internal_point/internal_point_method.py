import numpy as np
from ...core.exceptions import InitialPositionError
from ...core.exceptions import ConditionsError
from .newton_method import UnconstrainedNewtonMethod
from . import approx_path
from copy import copy
from . import backtracking_line_search
from . import newton_stop_criteria

class InternalPointMethod:
    class log_barrier_oracle:
        def __init__(self, t):
            self.t = t
        
        def get_dimension(self):
            return self.oracle.get_dimension()

        def func(self, x):
            try:
                assert(x.shape[0] == self.get_dimension())
            except:
                raise(ConditionsError(
                    'x has wrong size'
                ))
            if not self.constraints.satisfy(x):
               return np.inf
            try:
                res = self.t * self.oracle.func(x) - np.sum(np.log(-self.F.dot(x) + self.g))
            except:
                raise(ConditionsError(
                    'x is not strictly feasible with the constraints'
                ))
            return(res)
        
        def first_derivative(self, x):
            try:
                assert(x.shape[0] == self.get_dimension())
            except:
                raise(ConditionsError(
                    'x has wrong size'
                ))
            if not self.constraints.satisfy(x):
                raise(ConditionsError(
                    'x is not feasible with the constraints'
                ))
            try:
                res = self.t * self.oracle.first_derivative(x) - self.F.T.dot(1./(self.F.dot(x) - self.g))
            except:
                raise(ConditionsError(
                    'x is not strictly feasible with the constraints'
                ))
            return(res)
        
        def second_derivative(self, x):
            try:
                assert(x.shape[0] == self.get_dimension())
            except:
                raise(ConditionsError(
                    'x has wrong size'
                ))
            if not self.constraints.satisfy(x):
                raise(ConditionsError(
                    'x is not feasible with the constraints'
                ))
            try:
                res = np.zeros((self.get_dimension(), self.get_dimension()))
                val = (1/(self.F.dot(x) - self.g)) ** 2
                for i in range(self.F.shape[0]):
                    s = np.reshape(self.F[i], (1, self.F[i].shape[0]))
                    res += s.T.dot(s) * val[i]
                
                res += self.t * self.oracle.second_derivative(x)
            except  :
                raise(ConditionsError(
                    'x is not feasible with the constraints'
                ))
            return(res)
            

    def __init__(self, uneq_lin_constraints, oracle):
        self.constraints = uneq_lin_constraints
        self.oracle = oracle
        self.log_barrier_oracle.oracle = oracle

        F, g = uneq_lin_constraints.parameters()
        self.log_barrier_oracle.F = F
        self.log_barrier_oracle.g = g
        self.const_dimenstion = F.shape[0]
        self.log_barrier_oracle.constraints = uneq_lin_constraints
    
    def set_init_log_barrier_coeff(self, t=1):
        self.t_0 = t
    
    def set_init_position(self, x):
        try:
            assert(self.constraints.satisfy(x))
        except:
            raise(InitialPositionError(
                'The initial point is wrong'
            ))
        self.pos = x
    
    def make(self, tol = 0.001, mul_c = 1.1):
        path = approx_path()
        path.Append(self.pos)

        t = self.t_0
        x = self.pos

        # formulation of current newton_task:
        while True:
            oracle_t = self.log_barrier_oracle(t)
            solver = UnconstrainedNewtonMethod(oracle_t)
            solver.set_init_position(x)
            solver.set_step_size(backtracking_line_search(oracle_t))
            res = solver.make(newton_stop_criteria(oracle_t))
            x = res.GetLastValue()
            path.Append(x)
            if self.const_dimenstion/t < tol:
                break
            
            t = t*mul_c
        
        return path


