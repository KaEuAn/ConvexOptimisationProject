from 

class NoequalLinearConstraints:
    '''
    This class describes a set with 
    noequal linear constraints, 
    that is the set is given as following:
    Set = {x | F * x <= g}, 
    where F is matrix of n*m size,
    x is m - vector
    g is n - vector
    '''
    def __init__(F, b):
        self.F = F
        self.b = b
    
    def projection(y, norm='fro'):
        '''
        this function returns
        x = \argmin_{x \in Set}(norm(x - y))
        '''
        # do something here

        