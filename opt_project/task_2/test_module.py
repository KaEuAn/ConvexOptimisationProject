import unittest
import numpy as np

class TestTaskTwoConstraints(unittest.TestCase):
    def test_simple(self):
        from .constraints import TaskTwoLinearConstraints as LC
        F = np.array([
            [-1, 1],
            [1, -1]
        ])
        g = np.array([1, 1])
        a = np.array([1, 1])
        b = 1
        c = np.array([1, 1])
        d = 1
        lc = LC(F, g, a, b, c, d)
        lc.initialise()
        y = np.array([1, 18])
        y = lc.projection(y)

if __name__ == '__main__':
    unittest.main()