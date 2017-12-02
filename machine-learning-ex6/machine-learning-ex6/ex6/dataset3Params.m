function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = [];

for a = 1:length(C)
  for b = 1:length(sigma)  
    model = svmTrain(X, y, C(a), @(x1, x2) gaussianKernel(x1, x2, sigma(b)));
    predictions = svmPredict(model, Xval);
    
    error = mean(double(predictions ~= yval));
    
    #We store the results
    #Add all the previous results to the previous row
    #Then on the same row (columns), add the current C, sigma and error
    results = [results; C(a), sigma(b), error];
  end  
end

#Find the value on the third column (The error) with the min() function
#Then pullout that value and the row index of that value into minError and minIndex
[minError, minIndex] = min(results(:,3)); 

C = results(minIndex,1);
sigma = results(minIndex,2);
    
% =========================================================================

end
