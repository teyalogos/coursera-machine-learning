function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    h = theta(1) + (X(:, 2) * theta(2)); %we use this hypothesis to separate the intercept (1) from the actual parameters multiplied by X
    %note: you forgot to multiply theta(2) with it's corresponding X value
    
    theta_0 = theta(1) - alpha * (1/m) * sum((h - y) .* X(:, 1)); %the X(:, 1) is to only use the intercept column for theta zero
    theta_1 = theta(2) - alpha * (1/m) * sum((h - y) .* X(:, 2)); %the X(:, 2) is to only use the parameters for the actual gradient descent algorithm
    
    theta = [theta_0; theta_1]; %put together theta_0 and theta_1 to theta
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
