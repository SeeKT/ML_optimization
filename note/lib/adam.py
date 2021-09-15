"""
Adam
"""

import autograd.numpy as np 
from autograd import grad as nabla
from .lib_optimization import Base_optimization

class Adam(Base_optimization):
    def __init__(self, maxiter, eps, rho_1=0.9, rho_2=0.999, const_d=1e-8, delta=1e-6):
        super().__init__(maxiter, delta)
        self.eps = eps 
        self.rho_1 = rho_1 
        self.rho_2 = rho_2 
        self.const_d = const_d 
    
    def gets_adam(self, func, x, s):
        """
        Update biased first moment estimate,
        s <- rho_1 s + (1 - rho_1)g
        """
        grd = nabla(func)(x).reshape(x.shape)
        return self.rho_1*s + (1 - self.rho_1)*grd
    
    def getr_adam(self, func, x, r):
        """
        Update biased second moment estimate,
        r <- rho_2 r + (1 - rho_2) g*g
        """
        grd = nabla(func)(x).reshape(x.shape)
        return self.rho_2*r + (1 - self.rho_2)*grd*grd 

    def getupd_adam(self, s, r, t):
        """
        Compute update
        """
        hat_s = s/(1 - self.rho_1**t)
        hat_r = r/(1 - self.rho_2**t)
        pos = np.sqrt(hat_r) + self.const_d
        tmp = hat_s/pos 
        return (-1)*self.eps*tmp 

    def adam(self, func, xinit):
        """
        Adam
        """
        x_lst = [xinit]
        s = np.zeros_like(xinit); r = np.zeros_like(xinit)  # initialize 1st and 2nd moment variables
        itr = self.maxiter; flag = 1
        ##### iteration #####
        for i in range(1, self.maxiter):
            curx = x_lst[-1]        # current x
            s = self.gets_adam(func, curx, s)
            r = self.getr_adam(func, curx, r)
            d = self.getupd_adam(s, r, i)
            e = 1.0
            newx = self.update_gradient(curx, e, d)
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
        