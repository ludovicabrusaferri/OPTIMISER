import tensorflow as tf

def logistic_log_likelihood(data, location, scale):
    log_likelihood = tf.reduce_sum(-tf.math.log(scale) - (data - location) / scale - 2 * tf.math.log(1 + tf.exp(-(data - location) / scale)))
    return log_likelihood

def sample_from_logistic_distribution(location, scale, num_samples):
    uniform_samples = tf.random.uniform((num_samples,), dtype=tf.float32)
    samples = location + scale * tf.math.log(1.0 / uniform_samples - 1.0)
    return samples

