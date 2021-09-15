"""
RMSProp
"""

import autograd.numpy as np 
from autograd import grad as nabla
from .lib_optimization import Base_optimization

class RMSProp(Base_optimization):
    def __init__(self, maxiter, eps, rho=0.9, const_d = 1e-6, delta=1e-6):
        super().__init__(maxiter, delta)
        self.eps = eps 
        self.rho = rho 
        self.const_d = const_d
    
    def getv_rmsprop(self, func, x, r):
        """
        get accumulate squared gradient
        """
        grd = nabla(func)(x).reshape(x.shape)
        return self.rho*r + (1 - self.rho)*grd*grd 
    
    def getupd_rmsprop(self, func, x, r):
        """
        Compute update
        """
        grd = nabla(func)(x).reshape(x.shape)
        pos = np.sqrt(self.const_d + r)
        tmp = self.eps/pos 
        return (-1)*tmp*grd 
    
    def getrate_rmsprop(self, func, x, i, d):
        """
        learning rate (note that the update of the learning rate in RMSProp contains getv and getupd)
        """
        return 1.0 
    
    def rmsprop(self, func, xinit):
        """
        RMSProp
        """
        return self.iteration(
            func, 
            self.getv_rmsprop,
            self.getupd_rmsprop,
            self.getrate_rmsprop,
            xinit
        )