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

    
    summation1 = h_sum(m, theta, X, y, 1);
    summation2 = h_sum(m, theta, X, y, 2);
    theta(1,1) = theta(1,1) - 0.01 * 1/m * summation1;
    theta(2,1) = theta(2,1) - 0.01 * 1/m * summation2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function sum = h_sum(m, theta, X, y, feature_rank)
    theta0 = theta(1,1);
    theta1 = theta(2,1);
    sum = 0;
    for i = 1:m
       h_of_x = X(i,1) * theta0 + X(i,2) * theta1;
       medium_sum = (h_of_x - y(i)) * X(i, feature_rank);
       sum = sum + medium_sum;
    end    
end 


