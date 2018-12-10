import numpy as np

#returns iteration_stop_criteria
class iteration_stop_crit(object):
    def __init__(self, its):
        self.num = its
        self.it = 0
    
    def __call__(self, *args, **kwargs):
        self.it += 1
        return(self.it >= self.num)

#returns xdiff_stop_criteria
class xdiff_stop_crit(object):
    def __init__(self, diff):
        self.previous = np.inf
        self.diff = diff
    
    def __call__(self, value, *args, **kwargs):
        result = np.linalg.norm(self.previous - value) < self.diff
        self.previous = value
        return result

#returns xdiff_stop_criteria
class ydiff_stop_crit(object):
    def __init__(self, diff):
        self.previous = np.inf
        self.diff = diff

    def __call__(self, value, func_value, *args, **kwargs):
        result = np.linalg.norm(self.previous - func_value) < self.diff
        self.previous = func_value
        return result