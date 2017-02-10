function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. the parameters. 

% some useful values
m = length(y); % number of training examples

% Initializing the variables that will be returned when done 
J = 0;
grad = zeros(size(theta));

% Computing the cost and the gradient of a particular choice of theta. J
% should be set to the cost.

J = 1 / m * sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + (lambda ./ (2 .* m)) .* sum(theta(2:end) .^2);


grad(1) = (1 / m) .* sum((sigmoid(X * theta) - y) .* X(:, 1));

% Note that in the following code we do not need to use 'sum' as this is an 
% integral part of a matrix*vector multiplication!
grad(2:end) = (1 / m) .* (X(:, 2:end)' * (sigmoid(X * theta) - y)) + lambda ./ m .* theta(2:end);

grad = grad(:);

end
