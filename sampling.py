import numpy as np

def sample_from_logistic_distribution(location, scale, number_of_samples, alpha=None, threshold=None):
    location = np.asarray(location)
    scale = np.asarray(scale)

    if alpha is None:
        # Sample from a logistic distribution
        samples = np.random.logistic(loc=location, scale=scale, size=number_of_samples)
    else:
        alpha = np.asarray(alpha)

        if threshold is None:
            # Sample from a mixture of logistic distributions
            components = len(alpha)

            samples = np.zeros(number_of_samples)
            component_indices = np.random.choice(components, size=number_of_samples, p=alpha)

            for i in range(components):
                component_mask = (component_indices == i)
                component_samples = np.random.logistic(loc=location[i], scale=scale[i], size=np.sum(component_mask))
                samples[component_mask] = component_samples

        else:
            # Sample from a discretised mixture logistic distribution
            threshold = np.asarray(threshold)
            components = len(alpha)

            samples = np.zeros(number_of_samples)
            component_indices = np.random.choice(components, size=number_of_samples, p=alpha)
            uniform_samples = np.random.uniform(size=number_of_samples)

            for i in range(components):
                component_mask = (component_indices == i)
                component_samples = location[i] + (scale[i] * np.log(uniform_samples[component_mask] / (1 - uniform_samples[component_mask])))
                samples[component_mask] = component_samples

            samples = np.where(samples < threshold, samples, threshold)

    sample = np.mean(samples, axis=0)

    return sample

