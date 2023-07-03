def sample_from_logistic_distribution(location, scale, number_of_samples, alpha=None, threshold=None):
    location = location.numpy()
    scale = scale.numpy()

    if alpha is None:
        # Sample from a logistic distribution

        samples = []

        for i in range(number_of_samples):
            samples.append(np.random.logistic(loc=location, scale=scale))

        samples = np.array(samples)
    else:
        alpha = alpha.numpy()

        if threshold is None:
            # Sample from a mixture of logistic distributions

            components = len(alpha)

            samples = []

            for i in range(number_of_samples):
                component_idx = np.random.choice(components, p=alpha)

                samples.append(np.random.logistic(loc=location[component_idx], scale=scale[component_idx]))

            samples = np.array(samples)
        else:
            # Sample from a discretised mixture logistic distribution

            threshold = threshold.numpy()

            components = len(alpha)

            samples = []

            for i in range(number_of_samples):
                component_idx = np.random.choice(components, p=alpha)

                uniform_samples = np.random.uniform()

                samples.append(location[component_idx] + (scale[component_idx] *
                                                          np.log(uniform_samples / (1 - uniform_samples))))

            samples = np.array(samples)

            samples = np.where(samples < threshold, samples, threshold)

    samples = tf.convert_to_tensor(samples, dtype=dtype)
    sample = tf.math.reduce_mean(samples, axis=0)

    return sample
