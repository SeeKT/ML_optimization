"""
Library of the optimization problem, Test functions for optimization.
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import numpy as np 
from math import e

class Test_function():
    def __init__(self, n = 2):
        self.n = n      # the dimension of the vector x
    
    def rastrigin(self, x):
        """
        Rastrigin function,
        f(x) = An + sum_{i = 1}^n [x_i^2 - A cos(2 pi x_i)], where A = 10, -5.12 <= x_i <= 5.12,
        global minimum: f(0, ..., 0) = 0
        """
        val = 10*self.n
        for i in range(self.n):
            val += x[i]**2 - 10*np.cos(2*np.pi*x[i])
        return val 
    
    def ackley(self, x):
        """
        Ackley function,
        f(x, y) = -20exp(-0.2 sqrt(0.5(x^2 + y^2))) - exp(0.5(cos2 pi x + cos 2 pi y)) + e + 20, -5 <= x, y <= 5,
        global minimum: f(0, 0) = 0
        """
        val = -20*np.exp(-0.2*(np.sqrt(0.5*(x[0]**2 + x[1]**2))))
        val += -np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))
        val += e + 20 
        return val 
    
    def sphere(self, x):
        """
        Sphere function,
        f(x) = sum_{i = 1}^n x_i^2, -infty <= x_i <= infty, 1 <= i <= n,
        global minimum: f(0, ..., 0) = 0
        """
        return sum(x*x)
    
    def rosenbrock(self, x):
        """
        Rosenbrock function,
        f(x) = sum_{i = 1}^{n - 1} [100(x_{i + 1} - x_i^2) + (1 - x_i)^2], -infty <= x_i <= infty, 1 <= i <= n,
        global minimum: f(1, ..., 1) = 0
        """
        val = 0 
        for i in range(self.n - 1):
            val += 100*(x[i + 1] - x[i]**2) + (1 - x[i])**2 
        return val 
    
    def beale(self, x):
        """
        Beale function,
        f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2, -4.5 <= x, y <= 4.5,
        global minimum: f(3, 0.5) = 0
        """
        val = (1.5 - x[0] + x[0]*x[1])**2 
        val += (2.25 - x[0] + x[0]*x[1]**2)**2 
        val += (2.625 - x[0] + x[0]*x[1]**3)**2 
        return val 
    
    def goldstein_price(self, x):
        """
        Goldstein-Price function,
        f(x, y) = [1 + (x + y + 1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)][30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)], -2 <= x, y <= 2,
        global minimum: f(0, -1) = 3
        """
        first_term = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        second_term = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
        return first_term*second_term
    
    def booth(self, x):
        """
        Booth function,
        f(x, y) = (x + 2y - 7)**2 + (2x + y - 5)**2, -10 <= x, y <= 10,
        global minimum: f(1, 3) = 0
        """
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2 
    
    def bukin(self, x):
        """
        Bukin function N.6,
        f(x, y) = 100 sqrt(|y - 0.01x^2|) + 0.01|x + 10|, -15 <= x <= -5, -3 <= y <= 3,
        global minimum: f(-10, 1) = 0
        """
        return 100*np.sqrt(abs(x[1] - 0.01*x[0]**2)) + 0.01*abs(x[0] + 10)
    
    def matyas(self, x):
        """
        Matyas function,
        f(x, y) = 0.26(x^2 + y^2) - 0.48xy, -10 <= x, y <= 10,
        global minimum: f(0, 0) = 0
        """
        return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]
    
    def levi(self, x):
        """
        Levi function N.13,
        f(x, y) = sin(3 pi x)^2 + (x - 1)^2 (1 + sin(3 pi y)^2) + (y - 1)^2 (1 + sin(2 pi y)^2), -10 <= x <= 10,
        global minimum: f(1, 1) = 0
        """
        val = np.sin(3*np.pi*x[0])**2 
        val += (x[0] - 1)**2 * (1 + np.sin(3*np.pi*x[1])**2)
        val += (x[1] - 1)**2 * (1 + np.sin(2*np.pi*x[1])**2)
        return val 
    
    def himmelblau(self, x):
        """
        Himmelblau's function,
        f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2, -5 <= x, y <= 5,
        global minimum:
            f(3.0, 2.0) = 0
            f(-2.805118, 3.131312) = 0
            f(-3.779310, -3.283186) = 0
            f(3.584428, -1.848126) = 0
        """
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 
    
    def three_hump_camel(self, x):
        """
        Three-hump camel function,
        f(x, y) = 2x^2 - 1.05 x^4 + x^6/6 + xy + y^2, -5 <= x, y <= 5, 
        global minimum: f(0, 0) = 0
        """
        return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2 
    
    def easom(self, x):
        """
        Easom function,
        f(x, y) = -cos(x) cos(y) exp(-((x - pi)^2 + (y - pi)^2)), -100 <= x, y <= 100,
        global minimum: f(pi, pi) = -1
        """
        first_term = -np.cos(x[0])
        second_term = np.cos(x[1])
        third_term = np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))
        return first_term*second_term*third_term
    
    def cross_in_tray(self, x):
        """
        Cross-in-tray function,
        f(x, y) = -0.0001 [|sin(x) sin(y) exp(|100 - sqrt(x^2 + y^2)/pi|) | + 1]^{0.1}, -10 <= x, y <= 10,
        global minimum:
            f(1.34941, -1.34941) = -2.06261
            f(1.34941, 1.34941) = -2.06261
            f(-1.34941, 1.34941) = -2.06261
            f(-1.34941, -1.34941) = -2.06261
        """
        val = abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi)
        pos = np.sin(x[0])*np.sin(x[1])*np.exp(val)
        return -0.0001*(abs(pos) + 1)**0.1
    
    def eggholder(self, x):
        """
        Eggholder function,
        f(x, y) = -(y + 47)sin(sqrt(|x/2 + (y + 47)|)) - x sin(sqrt(|x - (y + 47)|)), -512 <= x, y <= 512,
        global minimum:
            f(512, 404.2319) = -959.6407
        """
        first_term = (-1)*(x[1] + 47)*np.sin(abs(x[0]/2 + (x[1] + 47)))
        second_term = x[0]*np.sin(abs(x[0] - (x[1] + 47)))
        return first_term - second_term 
    
    def holder_table(self, x):
        """
        Holder table function,
        f(x, y) = - | sin(x) cos(y) exp(|1 - sqrt(x^2 + y^2)/pi|)|, -10 <= x, y <= 10,
        global minimum:
            f(8.05502, 9.66459) = -19.2085
            f(-8.05502, 9.66459) = -19.2085
            f(8.05502, -9.66459) = -19.2085
            f(-8.05502, -9.66459) = -19.2085
        """
        val = abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi)
        return (-1)*abs(np.sin(x[0])*np.cos(x[1])*np.exp(val))
    
    def mccormick(self, x):
        """
        McCormick function,
        f(x, y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1, -1.5 <= x <= 4, -3 <= y <= 4,
        global minimum: f(-0.54719, -1.54719) = -1.9133
        """
        return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
    
    def schaffer_n2(self, x):
        """
        Schaffer function N.2,
        f(x, y) = 0.5 + (sin(x^2 - y^2)^2 - 0.5) / [1 + 0.001(x^2 + y^2)]^2, -100 <= x, y <= 100,
        global minimum: f(0, 0) = 0
        """
        val = np.sin(x[0]**2 - x[1]**2)**2 - 0.5 
        pos = 1 + 0.001*(x[0]**2 + x[1]**2)
        return 0.5 + val/(pos**2)
    
    def schaffer_n4(self, x):
        """
        Schaffer function N.4,
        f(x, y) = 0.5 + (cos(sin(|x^2 - y^2|))^2 - 0.5) / (1 + 0.001(x^2 + y^2))^2, -100 <= x, y <= 100,
        global minimum:
            f(0, 1.25313) = 0.292579
            f(0, -1.25313) = 0.292579
        """
        val = np.cos(np.sin(abs(x[0]**2 - x[1]**2)))**2 - 0.5 
        pos = 1 + 0.001*(x[0]**2 + x[1]**2)
        return 0.5 + val/(pos**2)
    
    def styblinski_tang(self, x):
        """
        f(x) = (sum_{i = 1}^n x_i^4 - 16x_i^2 + 5x_1) / 2, -5 <= x_i <= 5, 1 <= i <= n,
        global minimum: -39.16617n < f(-2.903534, ..., -2.903534) < -39.16616n
        """
        val = 0 
        for i in range(self.n):
            val += x[i]**4 - 16*x[i]**2 + 5*x[i]
        return val/2 
