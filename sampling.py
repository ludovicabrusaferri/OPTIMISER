import numpy as np

def sampling_function(meas, mean, sigma, N_samples, alpha=None, threshold=None):
    if alpha is None:
        # Generate N_samples from a logistic distribution
        samples = np.random.logistic(loc=mean, scale=sigma, size=N_samples)
        return samples
        
    else:
        if threshold is None:
            # Sample from a mixture of logistic distributions
            component_means = mean  # Mean for each component
            component_sigmas = sigma  # Standard deviation for each component

            # Choose component based on alpha probabilities
            components = np.random.choice(len(component_means), size=N_samples, p=alpha)

            # Sample from chosen components
            samples = np.zeros(N_samples)
            for i in range(len(component_means)):
                component_indices = np.where(components == i)[0]
                component_samples = np.random.logistic(loc=component_means[i], scale=component_sigmas[i], size=len(component_indices))
                samples[component_indices] = component_samples

            return samples

