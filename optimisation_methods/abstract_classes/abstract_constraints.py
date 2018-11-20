import numpy as np
import sympy 
import itertools as itt

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
    
    def __init__(self, F, b):
        self.F = F
        self.b = b
    
    @staticmethod
    def no_constraints_projection(A, b, y):
        '''
        The function returns the projection of 
        y point to the linear subspace, which is 
        defined as follow:
        A x = b (*)
        where the projection is the point y^', such that 
        y^' = argmin(|y - y^'| where y^' satisfies (*))
        '''

        # check for the linear system A x = b is compatible

        assert(A.shape[1] == b.shape[0])

        A_rank = np.linalg.matrix_rank(A)
        A_augmented = np.append(A, b.reshape(b.shape[0], 1), axis=1)
        A_augmented_rank = np.linalg.matrix_rank(A_augmented)

        # Kronecker–Capelli theorem conditions:
        if A_rank == A_augmented_rank :
            # the system has solutions
            pass
        else :
            # the system has no solutions
            return(None)
        
        # Below we select numbers of the rows in the A matrix, 
        # which are independent 
        _, indep_rows_numbers = sympy.Matrix(A).T.rref()
        A_indep = A[indep_rows_numbers]
        b_indep = b[indep_rows_numbers]

        # Below we define and solve the final system
        '''
        1. Let's assume, that x_0 is the solution of the system:
        A x_0 = b

        2. So we can suppose, that we find projection of (y - x_0) vector
        on the linear subspace L defined as follow:
        A x = 0
        
        3. Let's define y_0 := (y - x_0) 

        4. Let's suppose, that y* is the solution of our task
        This means, that p = (y_0 - y*) vector orthogonal to L

        5. So, we can notice, that in view of rows a_1, ..., a_k 
        of A matrix are orthogonal to L and 
        <a_1, a_2 ... a_k> is the orthogonal component of L, so
        p \in <a_1, a_2 ... a_k>
        
        6. Let's suppose  {a_i_1, a_i_2, ... , a_i_m}, where m <= k ,is the 
        basis of <a_1, a_2 ... a_k>. 
        Define {a'_1, ... , a'_m } := {a_i_1, a_i_2, ... , a_i_m}

        7. Let's define A' := (a'_1, ... , a'_m) , where A' is the matrix
        and a'_i are rows of this matrix (In the code below A' = A_indep)
        Also let's define b' := (b_i_1, ... b_i_m) (In the code b' = b_indep)

        8. So p = alpha_1 * a'_1  + ... + alpha_m * a'_m
        In matrix symbols, p = A'.T * alpha

        9. Let.s notice, that y_0 - p = y*, 
        y* belongs to L, so 0 = A' y* = A' (y_0 - p) = 
        = A' y_0 - A' A'.T alpha = A' (y - x_0) - A' A'.T alpha =
        = A' y - b' - A' A'.T alpha

        10. So, we have got the formula:
        A' A'.T alpha = A' y - b'
        in the code it is equivalent to solving the 
        linear system:
          alpha = np.linalg.solve(Mat, vec)
        '''

        Mat = A_indep @ A_indep.T
        vec = A_indep @ y - b_indep
        alpha = np.linalg.solve(Mat, vec)

        res_vector = A_indep.T @ alpha
        res_point = y - res_vector
        return(res_point)
    
    def Satisfy(self, x):
        '''
        This method returns True,
        if x satisfies the constraints,
        otherwise it returns False
        '''
        assert(self.F.shape[1] == x.shape[0])
        return(np.min(self.b - self.F @ x) >= 0)

    
    def projection(self, y):
        '''
        this function returns
        x = \argmin_{x \in Set}(norm(x - y)), 
        where norm is the euclidean norm
        '''

        proj_satisf_constr = []

        for n in range(1, self.F.shape[0] + 1):
            # n is the size of planes combinations 
            # to project y on them

            combs = itt.combinations(
                range(self.F.shape[0]), n)
            
            # finding of the projections on
            # selected combinations
            projs = []
            for comb in combs:
                proj = no_constraints_projection(
                    self.F[comb], 
                    b[comb], 
                    y)
                projs.append(proj)
            
            # find projections that satisfy 
            # the constraints
            proj_satisf_constr = list(
                filter(self.Satisfy, projs))
            
            if proj_satisf_constr:
                # Success!
                # We have merely to select the 
                # minimal projection
                break
        
        if not proj_satisf_constr:
            # something have gone wrong!
            assert(1 == 0)
        
        # selection the point, which 
        # correstponded to minimal projection
        min_proj = proj_satisf_constr[0]
        for proj in proj_satisf_constr:
            if np.linalg.norm(y - proj) < np.linalg.norm(y - min_proj):
                min_proj = proj
        
        return min_proj
        


            

            

        


        

        