"""
The current script describs the Logistic Regression Model.

The Sigmoid  is the function which the model try to fit in the actual 
data. x variable corresponds to the instances of rho while x0 is 
the value of rho where the phase transition phenomenon occurs. s is 
used  as a parameter which affects the slope of the Sigmoid function.

The Error function is used to calculate the residual between the actual 
results and a possible solution. y corresponds to the actual results.

The Derivative function calculates the first order derivatives of the

Error function and it takes the form of a jacobian matrix.

The Phase function is used to obtain the  phase transitions for 
a fraction of success 60 %(c=0.6),70%(0.7) etc.

The for loop is used to locate the phase transitions for all the
instances of delta. For each iteration a different instance of delta
is taken into account from the leastsq function. In order to seed
the leastsq function with the initial set of values the idx locates
probability of success which is closer to 0.5 and then the correposnding
instance of rho is used at the initial set of values c0.
x_centroid is the instance of rho where the phase transition phenomenon
occurs. 
x_general is the instance of rho for the phase transitions when the 
fraction of success is equal to c.

"""
# import ############################################################ import #
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

Success=np.array([[0.6,0.1,0.1,0,0,0,0,0,0,0],[0.8,0.2,0,0,0,0,0,0,0,0],[1.000,0.800,0,0,0,0,0,0,0,0],[0.9,1,0.3,0,0,0,0,0,0,0],[1,1,0.5,0.1,0,0,0,0,0,0],[1,1,1,0.2,0,0,0,0,0,0],[1,1,1,0.6,0,0,0,0,0,0],[1,1,1,1,0.3,0,0,0,0,0],[1,1,1,1,1,0.5,0.1,0,0,0],[1,1,1,1,1,0.9,1,0.7,0.2,0.5]])

# Sigmoid ########################################################## Sigmoid #

def Sigmoid(c,x): 
    x0, s = c
    return 1./(1 + np.exp(-s*(x-x0)))


# Error ########################################################## Error #

    
def Error(c,x,y):       
    return y-Sigmoid(c,x)
    
# Derivative ########################################################## Derivative  #
   
    
def Derivative(c,x,y): 
    x0, s = c
    jacobi = np.zeros((x.size, 2))
    jacobi[:,0] = s*np.exp(-s*(x - x0))/(1 + np.exp(-s*(x - x0)))**2
    jacobi[:,1] = (-x + x0)*np.exp(-s*(x - x0))/(1 + np.exp(-s*(x - x0)))**2
    return jacobi
    
# Phase ########################################################## Phase #


def Phase(c,r):
    x0,s=r
    x=x0-(np.log((1-c)/c))/s
    return x
    
s=20

   
x_centroid=np.zeros((10,1))
x_general=np.zeros((10,1))
delta=np.linspace(0.1,1,10)
rho=np.linspace(0.1,1,10)

# leastsq ########################################################## leastsq #


for i in range(10):
    idx = np.argmin(np.abs(Success[i,:] - 0.5))
    c0 = (rho[idx], s)
    y_n=Success[i,:]
    res,q=leastsq(Error,c0,args=(delta,y_n),Dfun=Derivative)
    x_centroid[i]=res[0]
    c=0.8
    x_general[i]= Phase(c,res)
    
# plot ########################################################## plot #
   
    
plt.figure()
plt.subplot(2,1,1)
plt.plot(delta,x_centroid, 'b')
plt.ylabel('\\rho') 
plt.xlabel('\delta')
plt.subplot(2,1,2)
plt.plot(delta,x_general, 'g')
plt.ylabel('\\rho') 
plt.xlabel('\delta')
plt.show()