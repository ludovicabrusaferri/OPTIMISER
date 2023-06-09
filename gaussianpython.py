import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from scipy.stats import norm

dtype = tf.float32

#def discretized_gaussian_negative_log_likelihood(data, mu, sigma, M):
 #   data_i = np.clip(data, -M, M)  # Truncate data within the shifted boundaries
  #  log_likelihood = np.sum(-0.5 * ((data_i - mu)**2 / (2*sigma**2) + np.log(np.sqrt(2*np.pi)*sigma)))
   # return -log_likelihood

#def discrete_gaussian_sampling(mu, sigma, num_samples, M):
 #   epsilon = np.random.randn(num_samples)  # Generate samples from standard normal distribution
  #  samples = np.round(mu + sigma * epsilon)  # Apply rounding operation
   # samples = np.clip(samples, -M, M)  # Truncate values within the shifted boundaries
   # return samples

#def discretized_gaussian_negative_log_likelihood_loss(data, mu, sigma, Mlow, Mup):
 #   data_i = np.clip(data, Mlow, Mup)
  #  log_likelihood = tf.reduce_sum(-0.5 * ((data - mu)**2 / (2*sigma**2) + tf.math.log(tf.sqrt(2*np.pi)*sigma)))
   # return -log_likelihood

#def sample_from_discretized_gaussian_distribution(mu, sigma, num_samples, Mlow, Mup):
 #   epsilon = tf.random.normal(shape=(num_samples,))  # Generate samples from standard normal distribution
  #  samples = tf.round(mu + sigma * epsilon)  # Apply rounding operation
   # samples = tf.clip_by_value(samples, Mlow, Mup)  # Truncate values within the shifted boundaries
    #return samples


def sample_from_discretized_gaussian_distribution(mu, sigma, num_samples, Mlow, Mup):
    epsilon = tf.random.normal(shape=(1, num_samples))  # Generate samples from standard normal distribution
    samples = tf.round(mu + sigma * epsilon)  # Apply rounding operation
    
    # Initialize the counter for rejections outside truncation bounds
    num_rejections = tf.constant(0, dtype=tf.float32)
    
    # Check each sample and count the rejections
    for i in range(num_samples):
        if samples[0, i] < -Mlow or samples[0, i] > Mup:
            num_rejections += 1.0
    
    # Truncate the samples to the specified bounds
    samples = tf.maximum(samples, -Mlow)
    samples = tf.minimum(samples, Mup)
    
    # Compute the correction factor
    correction_factor = num_rejections / tf.cast(num_samples, dtype=tf.float32)
    
    return samples, correction_factor


def discretized_gaussian_negative_log_likelihood_loss(data, mu, sigma, num_samples, Mlow, Mup):
    samples, correction_factor = sample_from_discretized_gaussian_distribution(mu, sigma, num_samples, Mlow, Mup)
    eps = 1e-09
    
    # Truncate the data to the specified bounds
    data = tf.maximum(data, -Mlow)
    data = tf.minimum(data, Mup)
    
    # Calculate the log-likelihood
    log_likelihood = tf.reduce_sum(
        -0.5 * ((data - mu)**2 / (2 * sigma**2) + tf.math.log(tf.sqrt(2 * np.pi) * sigma))
    ) - tf.math.log(correction_factor + eps)
    
    return -log_likelihood






def gaussian_negative_log_likelihood_loss(y_true, y_pred_mean, y_pred_std_dev):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred_mean = tf.cast(y_pred_mean, dtype=tf.float32)
    y_pred_std_dev = tf.cast(y_pred_std_dev, dtype=tf.float32)

    loss = - tf.reduce_mean(
        tf.math.log(y_pred_std_dev) +
        0.5 * tf.square(tf.math.divide_no_nan(y_true - y_pred_mean, y_pred_std_dev)) +
        0.5 * tf.math.log(2 * tf.constant(math.pi, dtype=tf.float32))
    )
    loss = tf.cast(loss, dtype=tf.float32)
    return -loss


def sample_from_gaussian_distribution(mean, std_dev, num_samples):
    mean = tf.cast(mean, dtype=tf.float32)
    std_dev = tf.cast(std_dev, dtype=tf.float32)

    samples = mean + std_dev * tf.random.normal((num_samples,))
    samples = tf.cast(samples, dtype=tf.float32)

    return samples


#def logistic_negative_log_likelihood_loss(y_true, y_pred_location, y_pred_scale):
 #   y_true = tf.cast(y_true, dtype=tf.float32)
  #  y_pred_location = tf.cast(y_pred_location, dtype=tf.float32)
   # y_pred_scale = tf.cast(y_pred_scale, dtype=tf.float32)

    #loss = - tf.math.reduce_mean(((-tf.math.log(y_pred_scale)) - ((y_true - y_pred_location) / y_pred_scale)) -
    #                             (2.0 * tf.math.log(1.0 + tf.exp((-(y_true - y_pred_location)) / y_pred_scale))))
    #loss = tf.cast(loss, dtype=dtype)

    #return loss


#def sample_from_logistic_distribution(location, scale, number_of_samples):
  #  location = tf.cast(location, dtype=tf.float32)
 #   scale = tf.cast(scale, dtype=tf.float32)

   # samples = location + (scale * tf.math.log((1.0 / tf.random.uniform((number_of_samples,))) - 1.0))
    #samples = tf.cast(samples, dtype=dtype)

    #return samples


def main():
    # True parameters
    y_true_location = 80
    y_true_scale = 50
    number_of_samples = 32768
    number_of_iterations = 32768
    y_true_Mup = 200
    y_true_Mlow = 0
    # Generate true data
    y_true = sample_from_discretized_gaussian_distribution(y_true_location, y_true_scale, number_of_samples,y_true_Mlow,y_true_Mup)

    # Optimization to estimate the expected value
    initial_y_pred_location = 100
    initial_y_pred_scale = 20
    initial_y_pred_Mup = 1000
    initial_y_pred_Mlow = 1000 

    y_pred_location = tf.Variable(initial_y_pred_location, name="y_pred_location", trainable=True, dtype=dtype)
    y_pred_scale = tf.Variable(initial_y_pred_scale, name="y_pred_scale", trainable=True, dtype=dtype)
    y_pred_Mup = tf.Variable(initial_y_pred_Mup, name="y_pred_Mup", trainable=True, dtype=dtype)
    y_pred_Mlow = tf.Variable(initial_y_pred_Mlow, name="y_pred_Mlow", trainable=True, dtype=dtype)

    optimiser = tf.optimizers.Adam(amsgrad=True)

    for i in range(number_of_iterations):
        with tf.GradientTape() as tape:
            loss = discretized_gaussian_negative_log_likelihood_loss(y_true, y_pred_location, y_pred_scale, number_of_samples,y_pred_Mlow, y_pred_Mup)

        print("Iteration: {0}, Loss: {1}".format(str(i + 1), str(loss.numpy())))

        gradients = tape.gradient(loss, [y_pred_location, y_pred_scale, y_pred_Mlow, y_pred_Mup])
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimiser.apply_gradients(zip(gradients, [y_pred_location, y_pred_scale, y_pred_Mlow, y_pred_Mup]))

    y_pred_location = y_pred_location.numpy()
    y_pred_scale = y_pred_scale.numpy()

    # Generate predicted data using the predicted parameters
    estimated_data = sample_from_discretized_gaussian_distribution(y_pred_location, y_pred_scale, number_of_samples, y_pred_Mlow, y_pred_Mup)

    # Plot the true and predicted distributions
    num_bins = 50
    x = np.linspace(np.min(y_true), np.max(y_true), num_bins)
    true_pdf, _ = np.histogram(y_true, bins=x, density=True)
    estimated_pdf, _ = np.histogram(estimated_data, bins=x, density=True)

    plt.figure(figsize=(10, 8))
    plt.plot(x[:-1], true_pdf, 'r', linewidth=2, label="True Distribution")
    plt.plot(x[:-1], estimated_pdf, 'b', linewidth=2, label="Predicted Distribution")

    # Plot initial value
    initial_data = sample_from_discretized_gaussian_distribution(initial_y_pred_location, initial_y_pred_scale, number_of_samples, initial_y_pred_Mlow, initial_y_pred_Mup)
    initial_pdf, _ = np.histogram(initial_data, bins=x, density=True)
    plt.plot(x[:-1], initial_pdf, 'g--', linewidth=2, label="Initial Distribution")

    plt.legend(fontsize=12)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title("True and Predicted Logistic Distributions", fontsize=12)

    # Save the plot as a PNG image
    plt.savefig("distributions.png")

    # Print the estimated parameters
    print("True Location:", y_true_location)
    print("True Scale:", y_true_scale)
    print("Predicted Location:", y_pred_location)
    print("Predicted Scale:", y_pred_scale)


if __name__ == '__main__':
    main()
