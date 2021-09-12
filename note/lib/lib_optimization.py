"""
Library of the optimization problem, gradient methods.
"""

import autograd.numpy as np 
from scipy.linalg import norm 

class Base_optimization():
    def __init__(self, maxiter, delta=1e-6):
        self.maxiter = maxiter 
        self.delta = delta 
    
    def check_converge(self, func, x1, x2):
        """
        check whether or not converges
        
        Input:
            x1, x2: the value of x
        Output:
            True or False
        """
        if norm(func(x1) - func(x2)) < self.delta:
            return True 
        else:
            return False

    def check_diverge(self, x1, x2):
        """
        check whether or not diverges

        Input:
            x1, x2: the value of x
        Output:
            True or False
        """
        if norm(x1 - x2) > 1e+5:
            return True 
        else:
            return False 

    def update_gradient(self, curx, eps, d):
        """
        update x (gradient method),
        x^{k + 1} = x^k + eps_k d
        
        Input: 
            curx: current x
            eps: learning rate
            d: update vector
        Output:
            updated value
        """
        return curx + eps*d 
    
    def iteration(self, func, getupd, getrate, xinit):
        """
        iteration optimization method until converge

        Input:
            func: objective function, func(x)
            getupd: the function to get update vector, getupd(function, x)
            getrate: the function to get learning rate, getrate(func, curx, d)
            xinit: the initial value of x
        Output:
            list of x and #Iter
        """
        x_lst = [xinit]     # value list
        itr = self.maxiter; flag = 1 
        ##### iteration #####
        for i in range(1, self.maxiter):
            curx = x_lst[-1]        # current x
            d = getupd(func, curx)  # update vector
            eps = getrate(func, curx, d)  # learning rate
            newx = self.update_gradient(curx, eps, d)
            if self.check_converge(func, curx, newx) and flag == 1:
                print("Converged. #Iter = {0}".format(i))
                itr = i; flag = 0 
            if self.check_diverge(curx, newx):
                print("Diverge!")
                break 
            else:
                x_lst.append(newx)
        #####################
        return np.array(x_lst), itr 