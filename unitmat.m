
% Generate true data from logistic distribution
true_location = 10;
true_scale = 5;
num_samples = 1000;
true_data = sample_from_logistic_distribution(true_location, true_scale, num_samples);

% Optimization using fmincon to estimate location and scale
options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100);
initial_guess = [1, 1]; % Initial guess for location and scale
[estimates, ~] = fmincon(@(params) -logistic_likelihood(true_data, params(1), params(2)), ...
    initial_guess, [], [], [], [], [0.01, 0.01], [100, 100], [], options);
estimated_location = estimates(1);
estimated_scale = estimates(2);

% Plot the true and estimated distributions
x = linspace(min(true_data), max(true_data), 100);
true_pdf = exp(-(x - true_location) ./ true_scale) ./ (true_scale * (1 + exp(-(x - true_location) ./ true_scale)).^2);
estimated_pdf = exp(-(x - estimated_location) ./ estimated_scale) ./ ...
    (estimated_scale * (1 + exp(-(x - estimated_location) ./ estimated_scale)).^2);

figure;
set(gcf, 'Color', 'w');
set(gca, 'FontSize', 30);
plot(x, true_pdf, 'r', 'LineWidth', 2);
hold on;
plot(x, estimated_pdf, 'b--', 'LineWidth', 2);
hold off;
legend('True Distribution', 'Estimated Distribution', 'FontSize', 20);
xlabel('Data');
ylabel('Probability Density');
title('True and Estimated Logistic Distributions');

% Logistic distribution sampling function
function samples = sample_from_logistic_distribution(location, scale, num_samples)
    samples = location + scale * log(1 ./ rand(num_samples, 1) - 1);
end

% Logistic likelihood function
function log_likelihood = logistic_likelihood(data, location, scale)
    z = (data - location) ./ scale;
    log_likelihood = sum(-log(scale) - z - 2 * log(1 + exp(-z)));
end

