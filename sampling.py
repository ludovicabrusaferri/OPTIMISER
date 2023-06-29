import numpy as np
from scipy.special import expit

def generate_samples(alpha, beta, mixture_weights, num_samples):
    num_components = len(mixture_weights)

    samples = []
    for _ in range(num_samples):
        # Step 4: Generate random input variables
        random_input = np.random.randn(alpha.shape[1])  # Assuming standard normal distribution

        # Step 5: Calculate log-odds for each component
        log_odds = [np.dot(random_input, beta_comp.T) for beta_comp in beta]

        # Step 6: Convert log-odds to probabilities for each component using the logistic function
        probabilities = [expit(log_odds_comp) for log_odds_comp in log_odds]

        # Step 7: Assign samples to components based on mixture weights
        component = np.random.choice(num_components, p=mixture_weights)

        # Step 8: Generate samples for the assigned component
        sample = np.random.uniform() <= probabilities[component]
        samples.append(sample)

    return samples

