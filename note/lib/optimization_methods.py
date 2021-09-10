"""
Library of the optimization problem, Optimization methods.
"""

import numpy as np 
from autograd import grad as nabla
from autograd import hessian
from scipy.linalg import norm 

class Optimization_methods():
    def __init__(self, maxiter, eps=1e-6):
        self.maxiter = maxiter
        self.eps = eps
    
    def check_converge(self, x1, x2):
        """
        check whether or not converges
        
        Input:
            x1, x2: the value of x
        Output:
            True or False
        """
        if norm(x1 - x2) < self.eps:
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

    def update_gradient(self, curx, alpha, d):
        """
        update x (gradient method),
        x^{k + 1} = x^k + alpha_k d
        
        Input: 
            curx: current x
            alpha: learning rate
            d: update vector
        Output:
            updated value
        """
        return curx + alpha*d 
    
    def linear_search(self, func, curx, d, beta=0.1, gamma=1e-5):
        """
        linear search (Armijo's rule)

        Input:
            func: objective function
            curx: current x
            d: update vector
            beta, gamma: parameters
        Output:
            tk: learning rate
        """
        grd = nabla(func)(curx).reshape(curx.shape) # current gradient
        ##### Armijo #####
        lk = 1
        while True:
            lft = func(curx + beta**lk * d) - func(curx)
            rit = gamma * beta**lk * grd.T @ d 
            if lft > rit:
                lk += 1 
            else:
                tk = beta**lk 
                break 
        ##################
        return tk 

    def steepest_descent(self, func, xinit, beta=0.1, gamma=1e-5):
        """
        Steepest descent
        Input:
            func: objective function
            xinit: the initial value of x
        Output:
            list of x and #Iter
        """
        x_lst = [xinit]     # value list
        itr = self.maxiter; flag = 1
        ##### iteration #####
        for i in range(1, self.maxiter):
            curx = x_lst[-1]
            grd = nabla(func)(curx).reshape(curx.shape) # current gradient
            d = (-1)*grd    # update vector
            alpha = self.linear_search(func, curx, d, beta, gamma)  # learning rate
            newx = self.update_gradient(curx, alpha, d)
            if self.check_converge(curx, newx) and flag == 1:
                print("Converged. #Iter = {0}".format(i))
                itr = i; flag = 0
                break 
            if self.check_diverge(curx, newx):
                print("Diverge!")
                break 
            else:
                x_lst.append(newx)
        #####################
        return np.array(x_lst), itr 


