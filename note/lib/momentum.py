"""
Momentum
"""

import autograd.numpy as np 
from autograd import grad as nabla
from .lib_optimization import Base_optimization

class Momentum(Base_optimization):
    def __init__(self, maxiter, eps, alpha, delta=1e-6):
        super().__init__(maxiter, delta)
        self.eps = eps 
        self.alpha = alpha 
    
    def getv_momentum(self, func, x, v):
        """
        compute velocity update
        """
        grd = nabla(func)(x).reshape(x.shape)
        return self.alpha*v - self.eps*grd 

    def getupd_momentum(self, func, x, v):
        """
        get update vector for Momentum
        """
        return v 
    
    def getrate_momentum(self, func, x, i, d):
        """
        get learning rate for Momentum
        """
        return 1.0 
    
    def momentum(self, func, xinit):
        """
        Momentum method
        """
        return self.iteration(
            func, 
            self.getv_momentum, 
            self.getupd_momentum,
            self.getrate_momentum, 
            xinit)
