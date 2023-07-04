import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def logistic_log_likelihood(data, location, scale):
    data = tf.cast(data, dtype=tf.double)
    location = tf.cast(location, dtype=tf.double)
    scale = tf.cast(scale, dtype=tf.double)

    log_likelihood = tf.reduce_sum(-tf.math.log(scale) - (data - location) / scale - 2 * tf.math.log(1 + tf.exp(-(data - location) / scale)))
    return log_likelihood

def sample_from_logistic_distribution(location, scale, num_samples):
    uniform_samples = tf.random.uniform((num_samples,), dtype=tf.float32)
    samples = location + scale * tf.math.log(1.0 / uniform_samples - 1.0)
    return samples

# True parameters
true_location = 10
true_scale = 5
num_samples = 10000

# Generate true data
true_data = sample_from_logistic_distribution(true_location, true_scale, num_samples)

# Optimization to estimate the expected value
initial_guess = [5, 1]

def objective(params):
    location = params[0]
    scale = params[1]
    return -logistic_log_likelihood(true_data, location, scale)

bounds = [(0.01, 100), (0.01, 100)]  # Boundaries for location and scale

result = minimize(objective, initial_guess, bounds=bounds)
estimated_location, estimated_scale = result.x

# Generate estimated data using the estimated parameters
estimated_data = sample_from_logistic_distribution(estimated_location, estimated_scale, num_samples)

# Plot the true and estimated distributions
num_bins = 50
x = np.linspace(np.min(true_data), np.max(true_data), num_bins)
true_pdf, _ = np.histogram(true_data, bins=x, density=True)
estimated_pdf, _ = np.histogram(estimated_data, bins=x, density=True)

plt.figure(figsize=(10, 8))
plt.plot(x[:-1], true_pdf, 'r', linewidth=2, label='True Distribution')
plt.plot(x[:-1], estimated_pdf, 'b', linewidth=2, label='Estimated Distribution')

# Plot initial value
initial_data = sample_from_logistic_distribution(initial_guess[0], initial_guess[1], num_samples)
initial_pdf, _ = np.histogram(initial_data, bins=x, density=True)
plt.plot(x[:-1], initial_pdf, 'g--', linewidth=2, label='Initial Distribution')

# Plot first iteration
first_iteration_data = sample_from_logistic_distribution(result.x[0], result.x[1], num_samples)
first_iteration_pdf, _ = np.histogram(first_iteration_data, bins=x, density=True)
#plt.plot(x[:-1], first_iteration_pdf, 'm--', linewidth=2, label='First Iteration Distribution')

plt.legend(fontsize=12)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('True and Estimated Logistic Distributions', fontsize=14)

# Save the plot as a JPEG image
plt.savefig('distributions.jpg')

# Print the estimated parameters
print("Estimated Location:", estimated_location)
print("Estimated Scale:", estimated_scale)

