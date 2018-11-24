import numpy as np
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar
import abc

class GradientDescent:
    def __init__(self, func, dimension):
        self.func = func
        self.n = dimension
        self.pos = np.zeros(self.n)
    
    def set_init_position(self, x):
        self.pos = x
    
    @abc.abstractmethod
    def get_gradient(self, a):
        return
    
    @abc.abstractmethod
    def get_alpha(self):
        return

    def make_step(self):
        nablaF = self.get_gradient(self.pos)
        self.pos = self.pos - self.get_alpha() * nablaF
    
    def make(self, stop_criteria):
        while not stop_criteria:
            self.make_step()
        return self.pos
