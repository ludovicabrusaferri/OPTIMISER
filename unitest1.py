import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def TF_logistic_log_likelihood(meas, mean, scale):
    meas = tf.cast(meas, tf.float32)
    mean = tf.cast(mean, tf.float32)
    scale = tf.cast(scale, tf.float32)
    return tf.reduce_sum(-tf.math.log(scale) - tf.math.log(1 + tf.exp((meas - mean) / scale)))

class TestLogisticLikelihood(unittest.TestCase):
    def test_logistic_likelihood(self):
        # Generate test data
        np.random.seed(42)  # Set a random seed for reproducibility
        num_samples = 100
        true_mean = 2.0
        true_scale = 1.5
        measurements = np.random.logistic(loc=true_mean, scale=true_scale, size=num_samples)

        # Convert the measurements to a TensorFlow tensor of type tf.float64
        meas_tensor = tf.convert_to_tensor(measurements, dtype=tf.float64)

        # Create the logistic distribution with the true parameters
        logistic_dist = tfp.distributions.Logistic(loc=true_mean, scale=true_scale)

        # Compute the log-likelihood using the function under test
        log_likelihood = TF_logistic_log_likelihood(meas_tensor, true_mean, true_scale)

        # Compute the expected log-likelihood using the built-in log_prob() method of the distribution
        expected_log_likelihood = tf.reduce_sum(logistic_dist.log_prob(meas_tensor))

        # Check if the computed log-likelihood matches the expected log-likelihood
        self.assertAlmostEqual(log_likelihood.numpy(), expected_log_likelihood.numpy(), delta=1e-2)

if __name__ == '__main__':
    unittest.main()

