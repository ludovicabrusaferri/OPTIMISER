import numpy as np

def sampling_function(meas, model, N_samples, beta, alpha=None):
    if alpha is None:
        # Sample from a logistic distribution
        samples = np.random.logistic(scale=beta * model, size=N_samples)
    else:
        # Sample from a mixture logistic distribution
        component_probabilities = alpha * 1 / (1 + np.exp(-beta * model))
        component_indices = np.random.choice(len(alpha), size=N_samples, p=component_probabilities)
        samples = np.random.logistic(scale=beta[component_indices] * model[component_indices])

    return samples
