"""
AdaGrad
"""

import autograd.numpy as np 
from autograd import grad as nabla
from .lib_optimization import Base_optimization

class AdaGrad(Base_optimization):
    def __init__(self, maxiter, eps, const_d = 1e-7, delta=1e-6):
        super().__init__(maxiter, delta)
        self.eps = eps 
        self.const_d = const_d
    
    def getv_adagrad(self, func, x, r):
        """
        get accumulate squared gradient
        r <- r + g*g
        """
        grd = nabla(func)(x).reshape(x.shape)
        return r + grd*grd 

    def getupd_adagrad(self, func, x, r):
        """
        Compute update
        """
        grd = nabla(func)(x).reshape(x.shape)
        pos = self.const_d + np.sqrt(r)
        tmp = self.eps/pos 
        return (-1)*tmp*grd 
    
    def getrate_adagrad(self, func, x, i, d):
        """
        learning rate (note that the update of the learning rate in AdaGrad contains getv and getupd)
        """
        return 1
    
    def adagrad(self, func, xinit):
        """
        AdaGrad
        """
        return self.iteration(
            func, 
            self.getv_adagrad,
            self.getupd_adagrad,
            self.getrate_adagrad,
            xinit
        )