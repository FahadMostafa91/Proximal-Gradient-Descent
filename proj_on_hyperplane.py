# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:36:55 2022

@author: gmostafa
"""

import numpy as np

from ls_with_qr import ls_with_qr as ls
from min_norm import min_norm

#Algorithm for the projection of v onto a hyperplane (also called affine subspace).
def proj_on_hyperplane(C,d,v):
    x = min_norm(C,d)
    print(x)
    w = ls(C.T,v)
    print(w)
    m = np.matmul(C.T,w)
    P = v - m
    z = x + P
    return z

# test problems
C =  np.array([[2.,1.,1.,4.]])
d = np.array([[7.]])
v = np.array([[1.],[1.],[1.],[1.]])


# solution 
z = proj_on_hyperplane(C,d,v)
print(z)