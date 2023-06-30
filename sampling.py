import numpy as np

def sampling_function(meas, model, N_samples, beta, alpha=None, threshold=None):
    if alpha is None:
        # Sample from a logistic distribution
        samples = np.random.logistic(scale=beta * model, size=N_samples)
    else:
        if threshold is None:
            # Sample from a mixture logistic distribution
            component_indices = np.random.choice(len(alpha), size=N_samples, p=alpha)
            component_samples = np.random.logistic(scale=beta * model)
            samples = component_samples[component_indices]
        else:
            # Sample from a discretized mixture logistic distribution
            component_indices = np.random.choice(len(alpha), size=N_samples, p=alpha)
            component_probabilities = 1 / (1 + np.exp(-beta * model))
            component_samples = np.random.binomial(1, component_probabilities)
            samples = np.where(component_samples == 1, np.random.logistic(scale=beta * model), threshold)

    return samples
