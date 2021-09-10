"""
Library of the optimization problem, Optimization methods.
"""

import numpy as np 
from autograd import grad as nabla
from autograd import hessian

class Optimization_methods():
    def __init__(self, maxiter):
        self.maxiter = maxiter