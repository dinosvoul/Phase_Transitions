"""
       Logistic Regression Model section
The current python script is used to map the results obtained from the
run of AGAIN.py onto the Logistic Regression Model and make the 
comparison with the upper and lower bound introduced by Donoho and
 Tanner.
       Cross Polytope Section
The rho_delta variable contains the tabulated values rho(delta)=0.5 
for the instances of delta that we take into account when we conduct 
our experiment at AGAIN.py. The values are obtained manually from
the corresponding matlab file polytope.mat.
epsilon corresponds to the fraction of success.
The for loop calculates rho with respect the formula of Donoho Tanner
 described in Equation  2.1 of the documentation.  
"""

# import ############################################################ import #

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
# read the results ############################################################ read the results #

f=h5py.File('testbroadenrange.hdf5','r')
dset1=f['vi']
Success=dset1[...]
f.close()

# Logistic Regression Model ############################################################ Logistic Regression Model #

def Sigmoid(c,x): 
    x0, s = c
    return 1./(1 + np.exp(-s*(x-x0)))



    
def Error(c,x,y):       
    return y-Sigmoid(c,x)
    
   
    
def Derivative(c,x,y): 
    x0, s = c
    jacobi = np.zeros((x.size, 2))
    jacobi[:,0] = s*np.exp(-s*(x - x0))/(1 + np.exp(-s*(x - x0)))**2
    jacobi[:,1] = (-x + x0)*np.exp(-s*(x - x0))/(1 + np.exp(-s*(x - x0)))**2
    return jacobi


def Phase(c,r):
    x0,s=r
    x=x0-(np.log((1-c)/c))/s
    return x
    
s=20
   
x_centroid=np.zeros((len(Success),1))
x_general=np.zeros((len(Success),1))
delta=np.linspace(0.05,1,20)
rho=np.linspace(0.05,1,20)
for i in range(len(Success)):    
    idx = np.argmin(np.abs(Success[i,:] - 0.5))
    y_n=Success[i,:]
    c0 = (rho[idx], s)
    res,q=leastsq(Error,c0,args=(delta,y_n),Dfun=Derivative,maxfev=1000)    
    x_centroid[i]=res[0]
    c=0.8
    x_general[i]= Phase(c,res)

# Cross Polytope ############################################################ Cross Polytope  #
   
    
N=1000
epsilon=0.5
contents = sio.loadmat('polytope.mat')
W_crosspolytope=contents.get('rhoW_crosspolytope')

count=0
rho_delta=[0.154,0.190,0.218,0.243,0.267,0.291,0.314,0.337,0.361,0.386,0.411,0.438,0.467,0.499,0.534,0.573,0.620,0.678,0.759,0.963]
delta_range=np.linspace(0.05,1,20)
matrix=np.zeros((20,1))
for delt in np.linspace(0.05,1,20):
    n= delt*N
    n= int(round(n))    
    s=(4*((N+2)**6))/epsilon
    p=np.log10(s)
    R=2*np.sqrt((n**(-1))*np.log10(s))
    k=n*rho_delta[count]*(1-R)
    matrix[count,:]=float(k/n)
    count+=1

# Plot ############################################################ Plot  #

    
plt.figure()
plt.rc('text', usetex=True)
plt.plot (W_crosspolytope[:,0], W_crosspolytope[:,1] , 'r')
plt.plot(delta_range,matrix,'bx')
plt.plot(delta_range,x_centroid, 'rx')
plt.line = plt.plot(W_crosspolytope[:,0], W_crosspolytope[:,1], label='Infinite Bound Cross Polytope')
plt.line = plt.plot(delta_range,x_centroid, label='Logistic Regression Model')
plt.line = plt.plot(delta_range,matrix, label='Finite Bound Cross Polytope')
plt.ylabel('\\rho') 
plt.xlabel('\delta')
plt.legend(loc='upper left')
plt.axis([0,1,0,1])
plt.grid()
plt.show()
