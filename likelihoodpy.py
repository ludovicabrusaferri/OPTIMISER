# Author: Ludovica Brusaferri
import os
import numpy as np
import tensorflow as tf 

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


def scipy_logistic_log_likelihood(meas, loc, scale):
    return np.sum(logistic.logpdf(meas, loc=loc, scale=scale))


def numpy_logistic_log_likelihood(meas, mean, scale):
    z = (meas - mean) / scale
    log_likelihood = np.sum(-np.log(scale) - z - 2 * np.log(1 + np.exp(-z)))
    return log_likelihood


def numpy_mixture_log_likelihood(meas, weights, means, scales):
    log_likelihoods = np.zeros_like(meas)
    for i in range(len(weights)):
        z = (meas - means[i]) / scales[i]
        log_likelihoods += weights[i] * (-np.log(scales[i]) - z - 2 * np.log(1 + np.exp(-z)))
    return np.sum(log_likelihoods)


# Example usage
meas = np.array([[1, 2, 3], [3, 5, 6]])
model = np.array([[0.1, 0.2, 0.3], [0.1, 0.5, 0.6]])
beta = np.array([[1, 2, 3], [4, 5, 6]])
alpha = np.array([[1, 2, 3], [4, 5, 6]])

result = discretized_logistic_mixture_log_likelihood(meas, model, beta, alpha)
print("Discretized Mixture Model Log Likelihood:", result)

meas_tensor = tf.convert_to_tensor(meas, dtype=tf.float32)
model_tensor = tf.convert_to_tensor(model, dtype=tf.float32)
beta_tensor = tf.convert_to_tensor(beta, dtype=tf.float32)
alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)

result = TF_discretized_logistic_mixture_log_likelihood(meas_tensor, model_tensor, beta_tensor, alpha_tensor)
print("TF Discretized Mixture Model Log Likelihood:", result.numpy())
































