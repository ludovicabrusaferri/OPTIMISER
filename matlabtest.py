cd('/Users/luto/Desktop/TEST2')
seed = 123;  % Set the desired seed value
rng(seed);  % Set the seed for the random number generator
% Generate true data from logistic distribution
true_mean = 20;
true_std_dev = 30;
num_samples = 10000;
true_Mlow=20;
true_Mup = 200; % True boundaries for the discrete Gaussian distribution
true_data = discreteGaussianSampling(true_mean, true_std_dev, num_samples, true_Mlow,true_Mup);

% Optimization using fmincon to estimate location, scale, and threshold
options = optimoptions('fmincon', 'Display', 'off', 'MaxIterations', 100);
initial_guess = [30, 21, 100,200]; % Initial guess for location, scale, and threshold (random guess)


% Estimate at the first iteration using initial guess
estimated_mean_initial = initial_guess(1);
estimated_std_dev_initial = initial_guess(2);
estimated_Mlow_initial = initial_guess(3); % Initial guess for M
estimated_Mup_initial = initial_guess(4); % Initial guess for M
estimated_data_initial = discreteGaussianSampling(estimated_mean_initial, estimated_std_dev_initial, num_samples, estimated_Mlow_initial, estimated_Mup_initial);

% Save all estimates
all_estimates = zeros(100, 4);
all_estimates(1, :) = initial_guess;

for i = 1:100
    [estimates, ~] = fmincon(@(params) -discretizedGaussianLogLikelihood(true_data, params(1), params(2), num_samples, params(3),params(4)), ...
        initial_guess, [], [], [], [], [0.01, 0.01, 0.01,0.1], [1000, 1000, 2000,2000], [], options);
    estimated_location = estimates(1);
    estimated_scale = estimates(2);
    estimated_Mlow = estimates(3);
    estimated_Mup = estimates(4);
    
    all_estimates(i, :) = [estimated_location, estimated_scale, estimated_Mlow,estimated_Mup];
    
    initial_guess = estimates; % Update initial guess for the next iteration
    fprintf("%d\n",i)
end

% Estimate the PDF using the sampling function
num_bins = 50;
x = linspace(min(true_data), max(true_data), num_bins);
true_pdf = histcounts(true_data, x, 'Normalization', 'pdf');
estimated_data_last_iteration = discreteGaussianSampling(estimated_location, estimated_scale, num_samples, estimated_Mlow,estimated_Mup);
estimated_pdf_last_iteration = histcounts(estimated_data_last_iteration, x, 'Normalization', 'pdf');
estimated_data_first_iteration = discreteGaussianSampling(estimated_mean_initial, estimated_std_dev_initial, num_samples, estimated_Mlow_initial, estimated_Mup_initial);
estimated_pdf_first_iteration = histcounts(estimated_data_first_iteration, x, 'Normalization', 'pdf');

% Plot the true and estimated distributions
figure('Visible', 'on');
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
saveas(gcf, 'distributions.jpg', 'jpg');

% Plot the true and estimated data
%figure('Visible', 'on');
%set(gcf, 'Color', 'w');
%set(gca, 'FontSize', 30);
%histogram(true_data, num_bins, 'Normalization', 'pdf', 'FaceColor', 'r', 'EdgeColor', 'none');
%hold on;
%histogram(estimated_data_last_iteration, num_bins, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'none');
%histogram(estimated_data_first_iteration, num_bins, 'Normalization', 'pdf', 'FaceColor', 'g', 'EdgeColor', 'none');
%hold off;
%legend('True Data', 'Estimated Data (Last Iteration)', 'Estimated Data (First Iteration)', 'FontSize', 20);
%xlabel('Data');
%ylabel('Probability Density');
%title('True and Estimated Logistic Distributions');
%saveas(gcf, 'distributions2.jpg', 'jpg');



function [samples, correction_factor] = discreteGaussianSampling(mu, sigma, num_samples, Mlow, Mup)
    epsilon = randn(1, num_samples);  % Generate samples from standard normal distribution
    samples = round(mu + sigma * epsilon);  % Apply rounding operation
    
    % Initialize the counter for rejections outside truncation bounds
    num_rejections = 0;
    
    % Check each sample and count the rejections
    for i = 1:num_samples
        if samples(i) < -Mlow || samples(i) > Mup
            num_rejections = num_rejections + 1;
        end
    end
    
    % Truncate the samples to the specified bounds
    samples = max(samples, -Mlow);
    samples = min(samples, Mup);
    
    % Compute the correction factor
    correction_factor = num_rejections / num_samples;
end

function log_likelihood = discretizedGaussianLogLikelihood(data, mu, sigma, num_samples, Mlow, Mup)
    [~, correction_factor] = discreteGaussianSampling(mu, sigma, num_samples, Mlow, Mup);
    eps = 1e-09;
    
    % Truncate the data to the specified bounds
    %data = max(data, -Mlow);
    %data = min(data, Mup);
    
    % Calculate the log-likelihood
    log_likelihood = sum(-0.5 * ((data - mu).^2 / (2 * sigma^2) + log(sqrt(2 * pi) * sigma))) - log(correction_factor + eps);
end
