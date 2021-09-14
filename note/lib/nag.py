"""
NAG
"""

import autograd.numpy as np 
from autograd import grad as nabla
from .lib_optimization import Base_optimization

class NAG(Base_optimization):
    def __init__(self, maxiter, eps, alpha, delta=1e-6):
        super().__init__(maxiter, delta)
        self.eps = eps 
        self.alpha = alpha 
    
    def interim_update_nag(self, x, v):
        """
        Apply interim update
        """
        return x + self.alpha*v 
    
    def estimate_gradient_nag(self, func, x, v):
        """
        Compute gradient estimate at interim point
        """
        xtilde = self.interim_update_nag(x, v)
        return nabla(func)(xtilde).reshape(x.shape)
    
    def getv_nag(self, func, x, v):
        """
        Compute velocity update
        """
        grd = self.estimate_gradient_nag(func, x, v)
        return self.alpha*v - self.eps*grd 
    
    def getupd_nag(self, func, x, v):
        """
        Compute theta upd
        """
        return v 
    
    def getrate_nag(self, func, x, i, v):
        """
        Compute learning rate
        """
        return 1
    
    def nag(self, func, xinit):
        """
        NAG
        """
        return self.iteration(
            func, 
            self.getv_nag,
            self.getupd_nag,
            self.getrate_nag,
            xinit
        )