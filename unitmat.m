% Generate true data from logistic distribution
true_location = 10;
true_scale = 5;
num_samples = 1000;
true_data = sample_from_logistic_distribution(true_location, true_scale, num_samples);

% Optimization using fmincon to estimate location and scale
options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100);
initial_guess = [30, 20]; % Initial guess for location and scale (random guess)

% Estimate at the first iteration using initial guess
estimated_location_initial = initial_guess(1);
estimated_scale_initial = initial_guess(2);
estimated_data_initial = sample_from_logistic_distribution(estimated_location_initial, estimated_scale_initial, num_samples);

% Save all estimates
all_estimates = zeros(100, 2);
all_estimates(1, :) = initial_guess;

for i = 1:100
    [estimates, ~] = fmincon(@(params) -logistic_log_likelihood(true_data, params(1), params(2)), ...
        initial_guess, [], [], [], [], [0.01, 0.01], [100, 100], [], options);
    estimated_location = estimates(1);
    estimated_scale = estimates(2);
    
    all_estimates(i, :) = [estimated_location, estimated_scale];
    
    initial_guess = estimates; % Update initial guess for the next iteration
end

% Estimate the PDF using the sampling function
num_bins = 50;
x = linspace(min(true_data), max(true_data), num_bins);
true_pdf = histcounts(true_data, x, 'Normalization', 'pdf');
estimated_data_last_iteration = sample_from_logistic_distribution(estimated_location, estimated_scale, num_samples);
estimated_pdf_last_iteration = histcounts(estimated_data_last_iteration, x, 'Normalization', 'pdf');
estimated_data_first_iteration = sample_from_logistic_distribution(estimated_location_initial, estimated_scale_initial, num_samples);
estimated_pdf_first_iteration = histcounts(estimated_data_first_iteration, x, 'Normalization', 'pdf');

% Plot the true and estimated distributions
figure;
set(gcf, 'Color', 'w');
set(gca, 'FontSize', 30);
plot(x(1:end-1), true_pdf, 'r', 'LineWidth', 2);
hold on;
plot(x(1:end-1), estimated_pdf_last_iteration, 'b', 'LineWidth', 2);
plot(x(1:end-1), estimated_pdf_first_iteration, 'g', 'LineWidth', 2);
hold off;
legend('True Distribution', 'Estimated Distribution (Last Iteration)', 'Estimated Distribution (First Iteration)', 'FontSize', 20);
xlabel('Data');
ylabel('Probability Density');
title('True and Estimated Logistic Distributions');

% Plot the true and estimated data
figure;
set(gcf, 'Color', 'w');
set(gca, 'FontSize', 30);
histogram(true_data, num_bins, 'Normalization', 'pdf', 'FaceColor', 'r', 'EdgeColor', 'none');
hold on;
histogram(estimated_data_last_iteration, num_bins, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'none');
histogram(estimated_data_first_iteration, num_bins, 'Normalization', 'pdf', 'FaceColor', 'g', 'EdgeColor', 'none');
hold off;
legend('True Data', 'Estimated Data (Last Iteration)', 'Estimated Data (First Iteration)', 'FontSize', 20);
xlabel('Data');
ylabel('Probability Density');
title('True and Estimated Data');

% Logistic log-likelihood


function log_likelihood = logistic_log_likelihood(data, location, scale)
    log_likelihood = sum(-log(scale) - (data - location) ./ scale - 2 * log(1 + exp(-(data - location) ./ scale)));
end

% Logistic distribution sampling function
function samples = sample_from_logistic_distribution(location, scale, num_samples)
    samples = location + scale * log(1 ./ rand(num_samples, 1) - 1);
end


