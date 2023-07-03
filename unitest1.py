import unittest
import numpy as np
from scipy.stats import logistic


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


class TestLogisticLikelihood(unittest.TestCase):
    def test_logistic_likelihood(self):
        # Generate test data
        np.random.seed(42)  # Set a random seed for reproducibility
        num_samples = 100
        true_mean = 2.0
        true_scale = 1.5
        measurements = np.random.logistic(loc=true_mean, scale=true_scale, size=num_samples)

        # Compute the log-likelihood using the numpy logistic function
        numpy_logistic_likelihood = numpy_logistic_log_likelihood(measurements, true_mean, true_scale)

        # Compute the log-likelihood using the numpy mixture function with weights=[1]
        numpy_mixture_likelihood = numpy_mixture_log_likelihood(measurements, [1], [true_mean], [true_scale])

        # Check if the numpy logistic log-likelihood matches the numpy mixture log-likelihood
        self.assertAlmostEqual(numpy_logistic_likelihood, numpy_mixture_likelihood, delta=1e-6)


if __name__ == '__main__':
    unittest.main()

