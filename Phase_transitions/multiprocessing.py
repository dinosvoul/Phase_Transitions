# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:57:40 2015

@author: DINOS
"""
import numpy as np
from pyunlocbox import functions
from pyunlocbox import solvers
from numpy import linalg as LA
import multiprocessing as mp
import mkl
mkl.set_num_threads(1)
tolerance=10**(-4)
N=100
Test=10
seed_rng = 1




def Experiment(iterable):   
    M=0
    Omega=np.random.permutation(N)
    x_0 = np.zeros(N)
    x_0[Omega[0:k]]=np.random.standard_normal(k)
    psi=np.ones(N)
    Psi=np.diag(psi)
    Phi= np.random.randn(n, N)
    A= np.dot(Phi,Psi)
    y= np.dot(A,x_0)
    x = np.zeros(N)
    f1 = functions.norm_l1()
    f2 = functions.proj_b2(epsilon=0, y=y, A=A, tight=False,\
    nu=np.linalg.norm(A, ord=2)**2)
    solver = solvers.douglas_rachford(step=1e-2)
    ret = solvers.solve([f1, f2], x, solver, rtol=1e-4, maxit=300)
    x=ret.get('sol')
    residual=(float(LA.norm(x-x_0,2))/LA.norm(x_0,2))**2
    if residual<=tolerance:
                M+=1
    return M
        
counter=-1
count=-1
Success= np.zeros((10,10))

for delta in np.linspace(0.1,1,10):
    n= delta*N
    n= int(round(n))
    count+=1
    for rho in np.linspace(0.1,1,10):
        k=rho*n
        k=int(round(k))
        counter+=1
        if __name__ == '__main__':
            pool = mp. Pool ( processes =10)
            iterable=[(n,k,i) for i in range(Test) ]              
            result = pool.map_async(Experiment,iterable).\
            get()
            Success[count,counter]=float(sum(result))/Test             
            pool.close()
            pool.join()                
    counter=-1
print(Success)


