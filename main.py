# Author: Ludovica Brusaferri
import os
import numpy as np

measures = np.random.rand(28,1)
model = np.random.rand(28,1)
sigma = np.random.rand(28,1)

def loglikelihood(prob):
     loglikelihood = np.log(prob)
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
    return np.log( sigma2 / sigma1 ) + ( np.power(sigma1,2) + np.power((mean1 - mean2),2) )/ (2 * np.power(sigma2,2)) - 1/2

def KLgaussianForMeanZeroAndStdOne(mean,sigma):
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # 𝐾𝐿(𝑝,𝑞)=log𝜎2𝜎1+𝜎21+(𝜇1−𝜇2)22𝜎22−12
    return np.log( 1 / sigma ) + ( np.power(sigma,2) + np.power((mean),2) )/ 2 - 1/2


def logistic_mixture_log_likelihood(meas,model,beta):
    # https://arunaddagatla.medium.com/maximum-likelihood-estimation-in-logistic-regression-f86ff1627b67
    return np.sum(meas*beta*model - np.log(1 + np.exp(beta*model)))

def logistic_mixture_log_likelihood(meas,model,beta,alpha):
    # https://arxiv.org/pdf/1802.10529.pdf
    # meas, mode, alpha and beta must have the same lenght
    return np.sum(np.sum(alpha*(np.sum(meas*beta*model - np.log(1 + np.exp(beta*model))))))
  
def discretized_logistic_mixture_log_likelihood(meas, model, beta, alpha, threshold=0.5):
    probs = 1 / (1 + np.exp(-(meas * beta * model)))
    y_pred = (probs >= threshold).astype(int)
    return np.sum(alpha * (np.log(probs) * y_pred + np.log(1 - probs) * (1 - y_pred)))

# Example usage
meas = np.array([[1, 2, 3], [4, 5, 6]])
model = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
beta = np.array([[1, 2, 3], [4, 5, 6]])
alpha = np.array([[1, 2, 3], [4, 5, 6]])

result = discretized_logistic_mixture_log_likelihood(meas, model, beta, alpha)
print("Discretized Mixture Model Log Likelihood:", result)
































