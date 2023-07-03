import unittest
import numpy as np
from scipy.stats import logistic

def scipy_logistic_log_likelihood(meas, loc, scale):
    return np.sum(logistic.logpdf(meas, loc=loc, scale=scale))

def numpy_logistic_log_likelihood(meas, mean, scale):
    z = (meas - mean) / scale
    log_likelihood = np.sum(-np.log(scale) - z - 2 * np.log(1 + np.exp(-z)))
    return log_likelihood

class TestLogisticLikelihood(unittest.TestCase):
    def test_logistic_likelihood(self):
        # Generate test data
        np.random.seed(42)  # Set a random seed for reproducibility
        num_samples = 100
        true_mean = 2.0
        true_scale = 1.5
        measurements = np.random.logistic(loc=true_mean, scale=true_scale, size=num_samples)

        # Compute the log-likelihood using the scipy function
        scipy_log_likelihood = scipy_logistic_log_likelihood(measurements, true_mean, true_scale)

        # Compute the log-likelihood using the numpy function
        numpy_log_likelihood = numpy_logistic_log_likelihood(measurements, true_mean, true_scale)

        # Check if the computed log-likelihoods match
        self.assertAlmostEqual(scipy_log_likelihood, numpy_log_likelihood, delta=1e-6)

if __name__ == '__main__':
    unittest.main()

