# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:48:10 2022

@author: gmostafa
"""


import numpy as np
from min_norm import min_norm
from ls_with_qr import ls_with_qr as ls


#===== Proximal_Gradiet method  ============# 
 
kmax = 1000
tol = 1e-6

def Problem(x,b):        # cost function

    y = np.matmul(x.T,x) + np.matmul(b.T,x)
    return float(y)

def proj_on_hyperplane(C,d,v):
    x = min_norm(C,d)
#    print(x)
    w = ls(C.T,v)
#    print(w)
    m = np.matmul(C.T,w)
    P = v - m
    z = x + P
    return z

def prox_grad_constrained_min(C,d,b,x,t):
    y = np.zeros((4,1)) 

    cost_current = Problem(x, b)
 
    k= 1
    diff = 2*tol
 #   q, r = np.linalg.qr(C.T)
    print('Initial cost value', cost_current)
    cost_current = Problem(x,b)
    while (diff > tol and k < kmax):
        y = x -t*(2*x+b)
        x = proj_on_hyperplane(C,d,y)
        cost_old = cost_current
        cost_current = Problem(x,b)
        k = k+1
        diff = abs(cost_old - cost_current)
    return x



if __name__=='__main__':
    C = np.array([[2.,1,1,4],[1,1,2,1]])
    x = np.array([[0.],[0.],[0.],[0.]])
    d = np.array([[7.],[6]])
#    v = np.array([[1.],[1.],[1.],[1.]])

    
# =============================================================================
    b = np.array([[-2.],[0.],[0.],[-3.]])
    t = 0.5
    x = prox_grad_constrained_min(C,d,b,x,t)
    print('optimal solution:',x)
    print('optimal function value:', Problem(x,b))
    print('constraint satisfaction:', np.matmul(C,x))
# =============================================================================
