import numpy as np

#returns iteration_stop_criteria
class iteration_stop_crit(object):
    def __init__(self, its):
        self.num = its
        self.it = 0
    
    def __call__(self):
        self.it += 1
        return(self.it >= self.num)

#returns xdiff_stop_criteria
class xdiff_stop_crit(object):
    def __init__(self, diff):
        self.previous = np.inf
        self.diff = diff
    
    def __call__(self, x):
        result = np.linalg.norm(self.previous - x) < self.diff
        self.previous = x
        return result

#returns xdiff_stop_criteria
class ydiff_stop_crit(object):
    def __init__(self, diff):
        self.previous = np.inf
        self.diff = diff

    def __call__(self, val):
        result = np.linalg.norm(self.previous - val) < self.diff
        self.previous = val
        return result