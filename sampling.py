import numpy as np
from scipy.special import expit

def generate_samples(beta, alpha, num_samples):
    num_components, num_features = beta.shape

    samples = []
    for _ in range(num_samples):
        # Step 1: Assign samples to components based on mixture weights
        component = np.random.choice(num_components, p=alpha)

        # Step 2: Generate random input variables
        random_input = np.random.randn(num_features)  # Assuming standard normal distribution

        # Step 3: Calculate log-odds for the selected component
        log_odds = np.dot(random_input, beta[component])

        # Step 4: Convert log-odds to probabilities using the logistic function
        probability = expit(log_odds)

        # Step 5: Generate samples based on the probability
        sample = np.random.uniform() <= probability
        samples.append(sample)

    return samples

