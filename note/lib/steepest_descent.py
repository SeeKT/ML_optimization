"""
Steepest descent method
"""

from autograd import grad as nabla
from .lib_optimization import Base_optimization

class Steepest(Base_optimization):
    def __init__(self, maxiter, delta=1e-6, beta=0.1, gamma=1e-5):
        super().__init__(maxiter, delta)
        self.beta = beta 
        self.gamma = gamma 
    
    def getupd_steepest(self, func, x):
        """
        get update vector for steepest method
        d = (-1)*grad
        """
        grd = nabla(func)(x).reshape(x.shape)
        return (-1)*grd 
    
    def getrate_steepest(self, func, x, d):
        """
        get learning rate for steepest method
        """
        grd = nabla(func)(x).reshape(x.shape)   # current gradient
        ##### Armijo #####
        lk = 1 
        while True:
            lft = func(x + self.beta**lk * d) - func(x)
            rit = self.gamma * self.beta**lk * grd.T @ d 
            if lft > rit:
                lk += 1
            else:
                tk = self.beta**lk 
                break 
        ##################
        return tk 
    
    def steepest_descent(self, func, xinit):
        """
        Steepest descent
        """
        return self.iteration(func, self.getupd_steepest, self.getrate_steepest, xinit)
        