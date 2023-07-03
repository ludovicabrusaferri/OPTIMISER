import unittest
import numpy as np
import tensorflow as tf
from scipy.stats import logistic


def scipy_logistic_log_likelihood(meas, loc, scale):
    return np.sum(logistic.logpdf(meas, loc=loc, scale=scale))


def tf_logistic_log_likelihood(meas, mean, scale):
    z = (meas - mean) / scale
    log_likelihood = tf.reduce_sum(-tf.math.log(tf.cast(scale, tf.float64)) - z - 2 * tf.math.log(1 + tf.exp(-z)))
    return log_likelihood


def tf_mixture_log_likelihood(meas, weights, means, scales):
    log_likelihoods = tf.zeros_like(meas)
    for i in range(len(weights)):
        z = (meas - means[i]) / scales[i]
        log_likelihoods += weights[i] * (-tf.math.log(tf.cast(scales[i], tf.float64)) - z - 2 * tf.math.log(1 + tf.exp(-z)))
    return tf.reduce_sum(log_likelihoods)


class TestLogisticLikelihood(unittest.TestCase):
    def test_logistic_likelihood(self):
        # Generate test data
        np.random.seed(42)  # Set a random seed for reproducibility
        num_samples = 100
        true_mean = 2.0
        true_scale = 1.5
        measurements = np.random.logistic(loc=true_mean, scale=true_scale, size=num_samples)

        # Convert measurements to TensorFlow tensor
        meas_tensor = tf.constant(measurements, dtype=tf.float64)

        # Compute the log-likelihood using the TensorFlow logistic function
        tf_logistic_likelihood = tf_logistic_log_likelihood(meas_tensor, true_mean, true_scale)

        # Compute the log-likelihood using the TensorFlow mixture function with weights=[1]
        tf_mixture_likelihood = tf_mixture_log_likelihood(meas_tensor, [1.0], [true_mean], [true_scale])

        # Compute the log-likelihood using the scipy logistic function
        scipy_logistic_likelihood = scipy_logistic_log_likelihood(measurements, true_mean, true_scale)

        # Check if the TensorFlow logistic log-likelihood matches the TensorFlow mixture log-likelihood
        self.assertAlmostEqual(tf_logistic_likelihood.numpy(), tf_mixture_likelihood.numpy(), delta=1e-6)

        # Check if the TensorFlow logistic log-likelihood matches the scipy logistic log-likelihood
        self.assertAlmostEqual(tf_logistic_likelihood.numpy(), scipy_logistic_likelihood, delta=1e-6)


if __name__ == '__main__':
    unittest.main()

