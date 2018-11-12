import numpy as np
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar




class GradientDescent:
    #returns iteration_stop_criteria
    class iteration_stop_crit(object):
        def __init__(self, its):
            self.num = its
            self.it = 0
        def __call__(self):
            self.it += 1
            return self.it <= self.num

    #returns
    class xdiff_stop_crit(object):
        def __init__(self, diff):
            self.previous = np.inf
            self.diff = diff
        def __call__(self, value):
            result = abs(self.previous - value) < self.diff
            self.previous = value
            return result

    def __init__(self, func, dimension):
        self.func = func
        self.n = dimension
        self.pos = np.zeros(self.n)
    def set_init_position(self, x):
        self.pos = x
    def get_gradient(self, a):
        return 0
    def get_alpha(self, nablaF):
        return 0.001

    def make_step(self):
        nablaF = self.get_gradient(self.pos)
        self.pos -= self.get_alpha(nablaF) * nablaF
    
    def make(self, stop_criteria=iteration_stop_crit(1000)):
        while not stop_criteria:
            self.make_step()
        return self.pos