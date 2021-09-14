"""
Stochastic gradient descent
"""

import autograd.numpy as np 
from autograd import grad as nabla
from .lib_optimization import Base_optimization

class Stochastic_gradient(Base_optimization):
    def __init__(self, maxiter, eps_0, eps_tau, tau, delta=1e-6):
        super().__init__(maxiter, delta)
        self.eps_0 = eps_0 
        self.eps_tau = eps_tau 
        self.tau = tau 
    
    def getv_sgd(self, func, x, v):
        """
        get velocity for SGD,
        if not considered, return 0
        """
        return 0 

    def getupd_sgd(self, func, x, v):
        """
        get update vector for SGD
        """
        grd = nabla(func)(x).reshape(x.shape)
        return (-1)*grd 
    
    def getrate_sgd(self, func, x, i, d):
        """
        get learning rate for SGD
        """
        if i <= self.tau:
            alpha = i/self.tau 
            return (1 - alpha)*self.eps_0 + alpha*self.eps_tau 
        else:
            return self.eps_tau 
    
    def stochastic_gradient(self, func, xinit):
        """
        Stochastic gradient descent
        """
        return self.iteration(
            func, 
            self.getv_sgd, 
            self.getupd_sgd, 
            self.getrate_sgd, 
            xinit)