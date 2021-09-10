"""
Library of the optimization problem, Optimization methods.
"""

import numpy as np 
from autograd import grad as nabla
from autograd import hessian

class Optimization_methods():
    def __init__(self, maxiter):
        self.maxiter = maxiter
    
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

    def steepest_descent(self, func, curx, alpha, d, beta=0.1, gamma=1e-5):
        """
        Steepest descent
        """