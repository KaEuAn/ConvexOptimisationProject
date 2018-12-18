import unittest
import numpy as np
from .newton_method import UnconstrainedNewtonMethod
from .newton_method import ConstrainedNewtonMethod
from . import backtracking_line_search
from . import newton_stop_criteria
from . import constrained_newton_stop_criteria
from ...core.abstract_constraints import NoequalLinearConstraints
from .internal_point_method import InternalPointMethod

class TestUnconstrainedNewtonMethod(unittest.TestCase):
    class _test_oracle:
        def __init__(self):
            self.dimension = 2
            pass
        
        def func(self, x):
            return np.sum((x - np.array([1, 2])) ** 2)
        
        def first_derivative(self, x):
            return 2 * x - np.array([2, 4])
        
        def get_dimension(self):
            return(self.dimension)
        
        def second_derivative(self, x):
            res = np.array([[2, 0], [0, 2]])
            return res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def test_one(self):
        oracle = self._test_oracle()
        solver = UnconstrainedNewtonMethod(oracle)
        bls = backtracking_line_search(oracle)
        init_poss = np.array([[1, 2], [3, 4], [100, 99]])
        right_answer = np.array([1, 2])
        for init_pos in init_poss:
            solver.set_init_position(init_pos)
            solver.set_step_size(bls)
            res = solver.make(newton_stop_criteria(oracle))
            res_point = res.GetLastValue()
            self.assertEqual(tuple(res_point), tuple(right_answer))
    
class TestConstrainedNewtonMethod(unittest.TestCase):
    class _test_oracle:
        def __init__(self):
            self.dimension = 3
            pass
        
        def func(self, x):
            return np.sum((x - np.array([1, 2, 4])) ** 2)
        
        def first_derivative(self, x):
            return 2 * x - np.array([2, 4, 8])
        
        def get_dimension(self):
            return(self.dimension)
        
        def second_derivative(self, x):
            res = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
            return res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def test_one(self):
        threshold = 1e-8
        oracle = self._test_oracle()
        A = np.array([[1, 1, 1]])
        b = np.array([-2])
        solver = ConstrainedNewtonMethod(oracle, A, b)
        bls = backtracking_line_search(oracle)
        init_poss = np.array([[0, 0, -2], [1, 2, -5], [100, 0, -102]])
        right_answer = np.array([-2, -1, 1])
        for init_pos in init_poss:
            solver.set_init_position(init_pos)
            solver.set_step_size(bls)
            res = solver.make(constrained_newton_stop_criteria(oracle))
            res_point = res.GetLastValue()
            self.assertTrue(np.linalg.norm(res_point - right_answer, np.inf) < threshold)

class TestInternalPointMethod(unittest.TestCase):
    class _test_oracle:
        def __init__(self):
            self.dimension = 3
            pass
        
        def func(self, x):
            return np.sum((x - np.array([1, 2, 4])) ** 2)
        
        def first_derivative(self, x):
            return 2 * x - np.array([2, 4, 8])
        
        def get_dimension(self):
            return(self.dimension)
        
        def second_derivative(self, x):
            res = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
            return res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_one(self):
        threshold = 1e-5
        constraints = NoequalLinearConstraints(np.array([[1, 3, 2], [7, -1, -3]]), np.array([16, -6]))
        oracle = self._test_oracle()
        solver = InternalPointMethod(constraints, oracle)
        init_poss = np.array([[-10, 0, 0], [-3, -3, 5], [0, 0, 3]])
        right_answer = np.array([1, 2, 4])
        for init_pos in init_poss:
            solver.set_init_log_barrier_coeff(t=1)
            solver.set_init_position(init_pos)
            res = solver.make(tol=0.0000001)
            res_point = res.GetLastValue()
            self.assertTrue(np.linalg.norm(res_point - right_answer, np.inf) < threshold)

if __name__ == '__main__':
    unittest.main()