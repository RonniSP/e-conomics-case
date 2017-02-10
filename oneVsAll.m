function [all_theta] = oneVsAll(X, y, num_labels, classes, lambda)
% ONEVSALL trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_theta, where the i-th row of all_theta 
% corresponds to the classifier for label i. Classes are the unique
% accounts
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);
logical_array = zeros(m,1);

% Initializing the variable that will be returned when done 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Training of the num_labels logistic regression classifiers with the 
% regularization parameter lambda.
% fmincg works similarly to fminunc, but is more efficient when dealing
% with large number of parameters.

for classifiers = 1:size(classes, 1)
    % Create a true/false vector that is true when y is equal to the
    % classifier
    current_class = classes(classifiers);
    logical_array = y == current_class;
    
    % Set Initial theta
    initial_theta = zeros(n + 1, 1);
    
    % Set options for fmincg
    options = optimset('GradObj', 'on', 'MaxIter', 50);

    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost 
    [theta] = fmincg(@(t)(lrCostFunction(t, X, logical_array, lambda)), initial_theta, options);
    
    all_theta(classifiers, :) = theta;
    
end


end
