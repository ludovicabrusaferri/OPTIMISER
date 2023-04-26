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
     temp = (meas - model) / sigma
     arr = - np.power(temp,2) / 2 - np.log(sigma) - np.log(2 * np.pi) / 2
     return np.sum(arr) / arr.shape[0]

def loglikelihoodregressionwitheps(meas,model,sigma,eps):
    sigma = sigma + eps
    temp = (meas - model) / sigma
    arr = - np.power(temp,2) / 2 - np.log(sigma) - np.log(2 * np.pi) / 2
    return np.sum(arr) / arr.shape[0]

def poissonloglikelihood(meas,model):
    arr = meas * np.log(model) - model
    return np.sum(arr) / arr.shape[0]

def KLgaussian(mean1,mean2,sigma1,sigma2):
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # 𝐾𝐿(𝑝,𝑞)=log𝜎2𝜎1+𝜎21+(𝜇1−𝜇2)22𝜎22−12
    return log( sigma2 / sigma1 ) + ( np.power(sigma1,2) + np.power((mean1-mean2),2) )/ 2*(np.power(sigma,2) ) - 1/2

def KLgaussianForMeanZeroAndStdOne(mean,sigma):
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # 𝐾𝐿(𝑝,𝑞)=log𝜎2𝜎1+𝜎21+(𝜇1−𝜇2)22𝜎22−12
    return log( 1 / sigma1 ) + ( np.power(sigma1,2) + np.power((mean1),2) )/ 2 - 1/2


value = loglikelihoodregression(measures,model,sigma)
print(value)




