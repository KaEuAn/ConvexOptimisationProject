from copy import deepcopy
from matplotlib import pyplot as plt
import time
import numpy as np


class ApproxPath(object):
    '''
    This class allows you to work with all optimization methods through the object 'ApproxPath'.
    This object is a sequence of points in which we were in the process of the method 
    Method 'Append(point)' adds a new point to the sequence 
    Method 'GetIterationAmount()' returns the number of iterations done in this method 
    Method 'Show(text)' draws a graph of the point dependence on the iteration number and calls it with the text
    Method 'GetSequence()' gives the sequence of points which we visited in the process of the optimization method 
    Method 'GetLastValue()' gives the last values of the sequence 
    The method should return this object as the result of its work
    '''
    def __init__(self):
        self.a = []
        self.time_marks = []
        self.has_exact_solut = False
    
    def InitializeExactSolution(self, x):
        '''
        This function initialize exact solution of 
        the problem
        '''
        self.has_exact_solut = True
        self.exact_solut = x
        
    def Append(self, point):
        self.a.append(point)
        self.time_marks.append(time.clock())
        
    def GetIterationAmount(self):
        return len(self.a)
    
    def Show(self, text=""):
        '''
        This function returns matplotlib graphic 
        with gradient descent track of each 
        vector component depending on the 
        iterations number
        '''
        iters = [i for i in range(len(self.a))]
        plt.figure(figsize=(15,7))
        plt.plot(iters, self.a)
        plt.grid(b=True, which='major', linestyle='-')
        plt.grid(b=True, which='minor', linestyle=':')
        plt.title(text, fontsize=20)
        plt.legend()
        plt.minorticks_on()
        plt.grid(True)
        plt.show()
        return None
    
    def ShowTime(self, text=""):
        '''
        This function returns matplotlib graphic
        with gradient descent error of the 
        current appropximation depending on 
        the time elapsed

        There are two variants of 
        error calculation:
        * if exact solution is defined, 
        so:
        err_{curr} = |x_{curr} - x_{exact}|

        * if exact solution is not defined,
        so:
        err_{curr} = |x_{curr} - x^*|,
        where x^* is the method result
        '''
        assert(len(self.time_marks) > 0)

        # times array generation
        t_0 = self.time_marks[0]
        times = [t - t_0 for t in self.time_marks]

        # errors array generation
        val_star = 0
        if self.has_exact_solut:
            val_star = self.exact_solut
        else:
            val_star = self.a[-1]
        
        errors = [np.linalg.norm(val- val_star) for val in self.a]

        plt.figure(figsize=(15,7))
        plt.plot(times, errors, 'k.:')
        plt.grid(b=True, which='major', linestyle='-')
        plt.grid(b=True, which='minor', linestyle=':')
        plt.xlabel(r"time elapsed, $sec$")
        plt.ylabel(r"error, euclidean norm")
        plt.title(text, fontsize=20)
        plt.legend()
        plt.minorticks_on()
        plt.grid(True)
        plt.show()
        return None
    
        
    def GetSequence(self):
        return deepcopy(self.a)
    
    def GetLastValue(self):
        return self.a[-1]
