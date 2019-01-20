function [ er, bad ] = NNTest(nn,test_x,test_y)
%NNTEST Summary of this function goes here
%   Detailed explanation goes here
% test_x = normalize(test_x, mu, sigma);
[er, bad] = nntest(nn, test_x, test_y);
end

