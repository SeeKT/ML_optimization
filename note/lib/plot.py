"""
Library of the optimization problem, Plot the contour and the surface of the function
"""

import numpy as np
import os 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

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


class Plot_func():
    def func_value(self, func, x_range, y_range):
        """
        get the value of the function (2 dim)
        Input:
            func: function
            x_range, y_range: numpy linspace
        Output:
            X, Y: numpy meshgrid
            Z: corresponding func value
        """
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.empty_like(X)
        for i in range(len(X)):
            for j in range(len(X[i])):
                vec_x = np.array([X[i][j], Y[i][j]])
                Z[i][j] = func(vec_x)
        return X, Y, Z
    
    def plot_contour(self, func, x_range, y_range, func_name):
        """
        plot the contour of the function (2 dim)
        """
        ##### get the value of the function #####
        X, Y, Z = self.func_value(func, x_range, y_range)
        #########################################

        ##### dir of fig #####
        base_dir = "./fig/"
        fig_dir = base_dir + "{0}/".format(func_name)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig_name = "contour_{0}".format(func_name)
        png_name = fig_dir + fig_name + ".png"
        eps_name = fig_dir + fig_name + ".eps"
        ######################

        ##### plot #####
        fig = plt.figure(figsize=(8, 5.25))
        fig.subplots_adjust(bottom=0.15, left=0.175)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
        ax.set_xlim([x_range[0], x_range[-1]]); ax.set_ylim([y_range[0], y_range[-1]])
        ax.contour(X, Y, Z, cmap='bwr')
        plt.show()
        #plt.savefig(png_name); plt.savefig(eps_name)
        ################
    
    def plot_surface(self, func, x_range, y_range, func_name):
        """
        plot the surface of the function
        """
        ##### get the value of the function #####
        X, Y, Z = self.func_value(func, x_range, y_range)
        #########################################

        ##### dir of fig #####
        base_dir = "./fig/"
        fig_dir = base_dir + "{0}/".format(func_name)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig_name = "surface_{0}".format(func_name)
        png_name = fig_dir + fig_name + ".png"
        eps_name = fig_dir + fig_name + ".eps"
        ######################

        ##### plot #####
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
        ax.set_xlim([x_range[0], x_range[-1]]); ax.set_ylim([y_range[0], y_range[-1]])
        surf = ax.plot_surface(X, Y, Z, cmap='bwr', linewidth=0)
        fig.colorbar(surf)
        plt.show()
        #plt.savefig(png_name); plt.savefig(eps_name)
        ################
    
    def plotly_surface_contour(self, func, x_range, y_range, func_name):
        """
        plot the surface of the function using plotly,
        https://plotly.com/python/3d-surface-plots/
        """
        ##### get the value of the function #####
        X, Y, Z = self.func_value(func, x_range, y_range)
        #########################################

        ##### dir of fig #####
        base_dir = "./fig/"
        fig_dir = base_dir + "{0}/".format(func_name)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig_name = "surface_{0}".format(func_name)
        html_name = fig_dir + fig_name + ".html"
        ######################

        ##### plot #####
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                            highlightcolor="limegreen", project_z=True))
        fig.update_layout(autosize=False,
                            scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                            width=500, height=500,
                            margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
        #fig.write_html(html_name)
        ################