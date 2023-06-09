import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


dtype = tf.float32


def logistic_negative_log_likelihood_loss(y_true, y_pred_location, y_pred_scale):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred_location = tf.cast(y_pred_location, dtype=tf.float32)
    y_pred_scale = tf.cast(y_pred_scale, dtype=tf.float32)

    loss = - tf.math.reduce_mean(((-tf.math.log(y_pred_scale)) - ((y_true - y_pred_location) / y_pred_scale)) -
                                 (2.0 * tf.math.log(1.0 + tf.exp((-(y_true - y_pred_location)) / y_pred_scale))))
    loss = tf.cast(loss, dtype=dtype)

    return loss


def sample_from_logistic_distribution(location, scale, number_of_samples):
    location = tf.cast(location, dtype=tf.float32)
    scale = tf.cast(scale, dtype=tf.float32)

    samples = location + (scale * tf.math.log((1.0 / tf.random.uniform((number_of_samples,))) - 1.0))
    samples = tf.cast(samples, dtype=dtype)

    return samples


def main():
    # True parameters
    y_true_location = 10.0
    y_true_scale = 2.0
    number_of_samples = 32768
    number_of_iterations = 32768

    # Generate true data
    y_true = sample_from_logistic_distribution(y_true_location, y_true_scale, number_of_samples)

    # Optimization to estimate the expected value
    initial_y_pred_location = 0.0
    initial_y_pred_scale = 1.0

    y_pred_location = tf.Variable(initial_y_pred_location, name="y_pred_location", trainable=True, dtype=dtype)
    y_pred_scale = tf.Variable(initial_y_pred_scale, name="y_pred_scale", trainable=True, dtype=dtype)

    optimiser = tf.optimizers.Adam(amsgrad=True)

    for i in range(number_of_iterations):
        with tf.GradientTape() as tape:
            loss = logistic_negative_log_likelihood_loss(y_true, y_pred_location, y_pred_scale)

        print("Iteration: {0}, Loss: {1}".format(str(i + 1), str(loss.numpy())))

        gradients = tape.gradient(loss, [y_pred_location, y_pred_scale])
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimiser.apply_gradients(zip(gradients, [y_pred_location, y_pred_scale]))

    y_pred_location = y_pred_location.numpy()
    y_pred_scale = y_pred_scale.numpy()

    # Generate predicted data using the predicted parameters
    estimated_data = sample_from_logistic_distribution(y_pred_location, y_pred_scale, number_of_samples)

    # Plot the true and predicted distributions
    num_bins = 50
    x = np.linspace(np.min(y_true), np.max(y_true), num_bins)
    true_pdf, _ = np.histogram(y_true, bins=x, density=True)
    estimated_pdf, _ = np.histogram(estimated_data, bins=x, density=True)

    plt.figure(figsize=(10, 8))
    plt.plot(x[:-1], true_pdf, 'r', linewidth=2, label="True Distribution")
    plt.plot(x[:-1], estimated_pdf, 'b', linewidth=2, label="Predicted Distribution")

    # Plot initial value
    initial_data = sample_from_logistic_distribution(initial_y_pred_location, initial_y_pred_scale, number_of_samples)
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
