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

def logistic_log_likelihood(meas, mean, scale):
    # https://arunaddagatla.medium.com/maximum-likelihood-estimation-in-logistic-regression-f86ff1627b67
    # Calculate the log-likelihood of the measurements given the logistic distribution
    return np.sum(-np.log(scale) - np.log(1 + np.exp((meas - mean) / scale)))

def TF_logistic_log_likelihood(meas, mean, scale):
    return tf.reduce_sum(-tf.math.log(scale) - tf.math.log(1 + tf.exp((meas - mean) / scale)))

    
def TF_misture_logistic_log_likelihood(meas, component_means, component_scales, alpha):
    # Calculate the log-likelihood of the measurements given the mixture of logistic distributions
    log_likelihoods = []
    for i in range(len(component_means)):
        log_likelihood_i = tf.reduce_sum(alpha[i] * (-tf.math.log(component_scales[i]) - tf.math.log(1 + tf.exp((meas - component_means[i]) / component_scales[i]))))
        log_likelihoods.append(log_likelihood_i)
    total_log_likelihood = tf.reduce_sum(log_likelihoods)
    return total_log_likelihood.numpy()


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
































