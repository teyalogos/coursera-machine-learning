function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Theta1 = Input Layer Theta2 = Hidden Layer

%a1 is X but with the hidden layer
a1 = [ones(m, 1) X];
%Compute sigmoid(X * Theta1). We are computing the input layer for Layer 2 (hidden layer)
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
%Compute Hidden Layer for the Output Layer
a3 = sigmoid(a2 * Theta2');
%imaxA3 is only the parts we want to pass on to p
%a3 is the final output of the Neural Network. maxA3 is the Hidden Layer and imaxA3 is the actual stuff we want (The Outputs).
[maxA3, imaxA3] = max(a3');
%p is equal to the predictions
p = imaxA3';

% =========================================================================


end
