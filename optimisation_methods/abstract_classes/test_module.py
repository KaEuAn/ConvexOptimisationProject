import unittest
import numpy as np

class TestConstraintsMethod(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestConstraintsMethod, self).__init__(*args, **kwargs)

        from abstract_constraints import NoequalLinearConstraints as LC
        self.LC = LC

class TestTwoDimensionalConstraints(TestConstraintsMethod):
    
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
        
    def test_simple_three(self):
        F = np.array([[1, 1], [-1, 1]])
        b = np.array([1, 1])
        y = np.array([2, 1])
        answer = np.array([1., 0.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))
    
    def test_simple_four(self):
        F = np.array([[1, 1], [-1, 1], [1, -1]])
        b = np.array([1, 1, 1])
        y = np.array([2, 1])
        answer = np.array([1., 0.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))
    
    def test_simple_five(self):
        F = np.array([[1, 1], [-1, 1], [1, -1]])
        b = np.array([1, 1, 1])
        y = np.array([0, 4])
        answer = np.array([0., 1.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y))) 

    def test_simple_six(self):
        F = np.array([[1, 1], [-1, 1], [1, -1]])
        b = np.array([1, 1, 1])
        y = np.array([0., 0.])
        answer = np.array([0., 0.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))
    
    def test_one(self):
        F = np.array([[-1/2, 1], [1/2, 1], [2, 1], [-4, 1], [1/3, -1]])
        b = np.array([1, 1, 4, 8, 3])
        y = np.array([-1, 5.5])
        answer = np.array([0., 1.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))
    
    def test_two(self):
        F = np.array([[-1/2, 1], [1/2, 1], [2, 1], [-4, 1], [1/3, -1]])
        b = np.array([1, 1, 4, 8, 3])
        y = np.array([-1., -2.])
        answer = np.array([-1., -2.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(answer), 
            tuple(lc.projection(y)))

    def test_three(self):
        F = np.array([[-1/2, 1], [1/2, 1], [2, 1], [-4, 1], [1/3, -1]])
        b = np.array([1, 1, 4, 8, 3])
        y = np.array([2.5, -5.5])
        answer = np.array([1.5, -2.5])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(np.round(answer, 6)), 
            tuple(np.round(lc.projection(y), 6)))
    
    def test_four(self):
        F = np.array([[-1/2, 1], [1/2, 1], [2, 1], [-4, 1], [1/3, -1]])
        b = np.array([1, 1, 4, 8, 3])
        y = np.array([4., -5])
        answer = np.array([3., -2.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(np.round(answer, 6)), 
            tuple(np.round(lc.projection(y), 6)))

    def test_five(self):
        F = np.array([[-1/2, 1], [1/2, 1], [2, 1], [-4, 1], [1/3, -1]])
        b = np.array([1, 1, 4, 8, 3])
        y = np.array([-4.5, -9])
        answer = np.array([-3., -4.])
        lc = self.LC(F, b)
        self.assertEqual(
            tuple(np.round(answer, 6)), 
            tuple(np.round(lc.projection(y), 6)))

if __name__ == '__main__' :
    unittest.main()

