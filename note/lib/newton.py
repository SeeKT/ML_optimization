"""
Newton method
"""

import autograd.numpy as np 
from autograd import grad as nabla
from autograd import hessian 
from .lib_optimization import Base_optimization

class Newton(Base_optimization):
    def __init__(self, maxiter, delta=1e-6):
        super().__init__(maxiter, delta)
    
    def getupd_newton(self, func, x):
        """
        get update vector for Newton method
        """
        grd = nabla(func)(x).reshape(x.shape)   # current gradient
        hesse = hessian(func)(x).reshape((x.size, x.size))
        return (-1)*np.linalg.inv(hesse) @ grd 
    
    def getrate_newton(self, func, x, i, d):
        """
        get learning rate for Newton
        """
        return 1.0 
    
    def newton(self, func, xinit):
        """
        Newton method
        """
        return self.iteration(func, self.getupd_newton, self.getrate_newton, xinit)