from copy import copy

class BackTrackingLineSearch:
    '''
    This class implements BackTracking Line Search method
    '''

    def __init__(self, oracle, dec_c=0.9, init_alpha=1, beta=0.0001, max_it = 20):
        self.oracle = oracle
        self.dec_c = dec_c
        self.init_alpha = init_alpha
        self.beta = beta
        self.max_it = max_it
    
    def __call__(self, x, val, grad_val, h):
        max_it = copy(self.max_it)
        f = self.oracle.func
        if val == None:
            val = f(x)
        if grad_val == None:
            grad_val = self.oracle.first_derivative(x)
        alpha = self.init_alpha
        while f(x + alpha * h) > val + self.beta * alpha * grad_val.dot(h) :
            if(max_it < 0) :
                print('BackTrackingLineSearch iterations exceeded')
                break
            alpha = self.dec_c * alpha
            max_it -= 0
        return alpha