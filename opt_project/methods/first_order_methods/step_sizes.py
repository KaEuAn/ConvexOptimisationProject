import numpy as np

class ConstantStepSize:
    def __init__(self, alpha) :
        self.alpha = alpha
    
    def __call__(self, val, grad_val, func):
        return self.alpha