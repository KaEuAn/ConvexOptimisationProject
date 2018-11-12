import numpy as np
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar

# returns stop_criteria_function 
def iteration_dec(it_num):
    num = it_num
    def decor(*args, **kwargs):
        iter += 1
        return iter > num
    return decor

class GradientDescent:
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
    
    def make(self, stop_criteria=iteration_dec(1000)):
        while not stop_criteria:
            self.make_step()
        return self.pos