# Author: Ludovica Brusaferri
import os
import numpy as np

measures = np.random.rand(28,1)
model = np.random.rand(28,1)
sigma = np.random.rand(28,1)

def loglikelihood(prob):
     loglikelihood = log(prob)
     return loglikelihood

def loglikelihoodregression(meas,model,sigma):
     var = (meas - model) / sigma
     arr = - np.power(var,2) / 2 - np.log(sigma) - np.log(2 * np.pi) / 2
     return np.sum(arr)

value = loglikelihoodregression(measures,model,sigma)
print(value)




