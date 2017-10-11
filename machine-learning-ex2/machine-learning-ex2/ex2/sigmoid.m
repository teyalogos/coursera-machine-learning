function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%denominator = 1 + exp(-1 * z);   %output = eulers number (2.72) ^ input (-1 * z or -z) 
%g = 1 ./ denominator;

                      %z = theta transpose x
d = 1 + exp(-1 * z);  %1 + e ^ -z
g = 1 ./ d;           %1 / 1 + e ^ -z

% =============================================================

end
