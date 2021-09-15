"""
Numerical experiment to try optimization methods
"""

import autograd.numpy as np 
from lib.lib import Test_function
from lib.plot import Plot_func
from lib.steepest_descent import Steepest
from lib.newton import Newton
from lib.momentum import Momentum
from lib.nag import NAG
from lib.adagrad import AdaGrad
from lib.rmsprop import RMSProp
from lib.adam import Adam
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import os 

########### plt ##########
plt.rcParams["font.family"] = "serif"       # fonts
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 15              
plt.rcParams["mathtext.cal"] = "serif"      
plt.rcParams["mathtext.rm"] = "serif"       
plt.rcParams["mathtext.it"] = "serif:italic"
plt.rcParams["mathtext.bf"] = "serif:bold"  
plt.rcParams["mathtext.fontset"] = "cm"     
##########################


class Plot_value(Plot_func):

    def plot_eachmethod(self, frame, ax, x_method, method_name):
        """
        plot each method
        """
        x_val = x_method.T[0]; y_val = x_method.T[1]
        l = len(x_val)
        if frame <= l:
            ax.plot(x_val[:frame], y_val[:frame], label=method_name)
            

    def update(self, frame, ax, x_steepest, x_newton, x_momentum, x_nag, x_adagrad, x_rmsprop, x_adam, x_range, y_range, X, Y, Z):
        """
        update function for an animation
        """
        ax.clear()
        ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
        ax.set_xlim([x_range[0], x_range[-1]]); ax.set_ylim([y_range[0], y_range[-1]])
        ax.set_title("Iter = {0}".format(frame))
        ax.contour(X, Y, Z, cmap='bwr')
        ##### plot each method #####
        self.plot_eachmethod(
            frame, ax, x_steepest, "steepest"
        )
        self.plot_eachmethod(
            frame, ax, x_newton, "Newton"
        )
        self.plot_eachmethod(
            frame, ax, x_momentum, "Momentum"
        )
        self.plot_eachmethod(
            frame, ax, x_nag, "NAG"
        )
        self.plot_eachmethod(
            frame, ax, x_adagrad, "AdaGrad"
        )
        self.plot_eachmethod(
            frame, ax, x_rmsprop, "RMSProp"
        )
        self.plot_eachmethod(
            frame, ax, x_adam, "Adam"
        )
        ax.legend(loc='upper left')
        ############################


    def plot_trajectory(self, func, x_range, y_range, x_steepest, x_newton, x_momentum, x_nag, x_adagrad, x_rmsprop, x_adam, itr, func_name):
        """
        plot the trajectory of x
        """
        ##### get the value of the function #####
        X, Y, Z = self.func_value(func, x_range, y_range)
        #########################################

        ##### dir of fig #####
        base_dir = "./fig/"
        fig_dir = base_dir + "{0}/all/".format(func_name)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig_name = "trajectory_{0}".format(func_name)
        gif_name = fig_dir + fig_name + ".gif"
        ######################

        ##### animation #####
        fig = plt.figure(figsize=(8, 5.25))
        fig.subplots_adjust(bottom=0.15, left=0.175)
        ax = fig.add_subplot(1,1,1)
        ani = animation.FuncAnimation(
            fig, 
            self.update, 
            fargs=(ax, x_steepest, x_newton, x_momentum, x_nag, x_adagrad, x_rmsprop, x_adam, x_range, y_range, X, Y, Z),
            frames=itr+1,
            interval=100
        )
        ani.save(gif_name, writer="Pillow")
        #####################




def main():
    """
    main function
    """
    ##### Parameters #####
    maxiter = 1000 
    ######################

    ##### instance of the classes #####
    test = Test_function()
    steepest = Steepest(maxiter=maxiter)
    newton = Newton(maxiter=maxiter)
    momentum = Momentum(maxiter=maxiter, eps=1e-2, alpha=0.9)
    nag = NAG(maxiter=maxiter, eps=1e-2, alpha=0.9)
    adagrad = AdaGrad(maxiter=maxiter, eps=0.1)
    rmsprop = RMSProp(maxiter=maxiter, eps=1e-2)
    adam = Adam(maxiter=maxiter, eps=1e-2)
    pv = Plot_value()
    ###################################

    
    ##### execute each iteration #####
    # (1) distorted function
    xinit = np.array([1.0, 1.5])
    distort_steepest_x, itr_distorted_steepest = steepest.steepest_descent(
        func=test.distorted, xinit=xinit
    )
    distort_newton_x, itr_distorted_newton = newton.newton(
        func=test.distorted, xinit=xinit
    )
    distort_momentum_x, itr_distorted_momentum = momentum.momentum(
        func=test.distorted, xinit=xinit
    )
    distorted_nag_x, itr_distorted_nag = nag.nag(
        func=test.distorted, xinit=xinit
    )
    distorted_adagrad_x, itr_distorted_adagrad = adagrad.adagrad(
        func=test.distorted, xinit=xinit
    )
    distorted_rmsprop_x, itr_distorted_rmsprop = rmsprop.rmsprop(
        func=test.distorted, xinit=xinit
    )
    distorted_adam_x, itr_distorted_adam = adam.adam(
        func=test.distorted, xinit=xinit
    )
    ##################################

    ##### plot #####
    x_range = np.linspace(-2.0, 2.0, 51); y_range = np.linspace(-2.0, 2.0, 51)
    iter_max = max(itr_distorted_steepest, itr_distorted_newton, itr_distorted_momentum, itr_distorted_nag, itr_distorted_adagrad, itr_distorted_rmsprop, itr_distorted_adam)
    pv.plot_trajectory(
        func=test.distorted,
        x_range=x_range,
        y_range=y_range,
        x_steepest=distort_steepest_x,
        x_newton=distort_newton_x,
        x_momentum=distort_momentum_x,
        x_nag=distorted_nag_x,
        x_adagrad=distorted_adagrad_x,
        x_rmsprop=distorted_rmsprop_x,
        x_adam=distorted_adam_x,
        itr=iter_max,
        func_name="distorted"
    )
    ################

    ##### execute each iteration #####
    # (2) Test function 2
    xinit = np.array([-0.25, 0.75])
    f2_steepest_x, itr_f2_steepest = steepest.steepest_descent(
        func=test.test_func_2, xinit=xinit
    )
    f2_newton_x, itr_f2_newton = newton.newton(
        func=test.test_func_2, xinit=xinit
    )
    f2_momentum_x, itr_f2_momentum = momentum.momentum(
        func=test.test_func_2, xinit=xinit
    )
    f2_nag_x, itr_f2_nag = nag.nag(
        func=test.test_func_2, xinit=xinit
    )
    f2_adagrad_x, itr_f2_adagrad = adagrad.adagrad(
        func=test.test_func_2, xinit=xinit
    )
    f2_rmsprop_x, itr_f2_rmsprop = rmsprop.rmsprop(
        func=test.test_func_2, xinit=xinit
    )
    f2_adam_x, itr_f2_adam = adam.adam(
        func=test.test_func_2, xinit=xinit
    )
    ##################################

    ##### plot #####
    x_range = np.linspace(-0.5, 1.5, 51); y_range = np.linspace(-0.5, 1.5, 51)
    iter_max = max(itr_f2_steepest, itr_f2_newton, itr_f2_momentum, itr_f2_nag, itr_f2_adagrad, itr_f2_rmsprop, itr_f2_adam)
    pv.plot_trajectory(
        func=test.test_func_2,
        x_range=x_range,
        y_range=y_range,
        x_steepest=f2_steepest_x,
        x_newton=f2_newton_x,
        x_momentum=f2_momentum_x,
        x_nag=f2_nag_x,
        x_adagrad=f2_adagrad_x,
        x_rmsprop=f2_rmsprop_x,
        x_adam=f2_adam_x,
        itr=iter_max,
        func_name="test_func_2"
    )
    ################


    ##### execute each iteration #####
    # (3) Test function 3
    xinit = np.array([-2.0, -2.0])
    f3_steepest_x, itr_f3_steepest = steepest.steepest_descent(
        func=test.test_func_3, xinit=xinit
    )
    f3_newton_x, itr_f3_newton = newton.newton(
        func=test.test_func_3, xinit=xinit
    )
    f3_momentum_x, itr_f3_momentum = momentum.momentum(
        func=test.test_func_3, xinit=xinit
    )
    f3_nag_x, itr_f3_nag = nag.nag(
        func=test.test_func_3, xinit=xinit
    )
    f3_adagrad_x, itr_f3_adagrad = adagrad.adagrad(
        func=test.test_func_3, xinit=xinit
    )
    f3_rmsprop_x, itr_f3_rmsprop = rmsprop.rmsprop(
        func=test.test_func_3, xinit=xinit
    )
    f3_adam_x, itr_f3_adam = adam.adam(
        func=test.test_func_3, xinit=xinit
    )
    ##################################

    ##### plot #####
    x_range = np.linspace(-6.0, 2.0, 51); y_range = np.linspace(-6.0, 2.0, 51)
    iter_max = max(itr_f3_steepest, itr_f3_newton, itr_f3_momentum, itr_f3_nag, itr_f3_adagrad, itr_f3_rmsprop, itr_f3_adam)
    pv.plot_trajectory(
        func=test.test_func_3,
        x_range=x_range,
        y_range=y_range,
        x_steepest=f3_steepest_x,
        x_newton=f3_newton_x,
        x_momentum=f3_momentum_x,
        x_nag=f3_nag_x,
        x_adagrad=f3_adagrad_x,
        x_rmsprop=f3_rmsprop_x,
        x_adam=f3_adam_x,
        itr=iter_max,
        func_name="test_func_3"
    )
    ################

if __name__ == "__main__":
    main()
