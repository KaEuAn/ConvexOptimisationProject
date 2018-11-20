import unittest
import numpy as np

class TestConstraintsMethod(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestConstraintsMethod, self).__init__(*args, **kwargs)

        from abstract_constraints import NoequalLinearConstraints as LC
        self.LC = LC
    
    def test_simple_one(self):
        F = np.array([[1, 1], [1, -1]])
        b = np.array([0, 0])
        y = np.array([3, 1])
        answer = np.array([0., 0.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))
    
    def test_simple_two(self):
        F = np.array([[1, 1], [-1, 1]])
        b = np.array([0, 0])
        y = np.array([3, 1])
        answer = np.array([1., -1.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))

if __name__ == '__main__' :
    unittest.main()

