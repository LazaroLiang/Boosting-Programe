function [model,error_rate,result] = threshold_tr(train_set, sample_weights, labels)
%
% TRAINING THRESHOLD CLASSIFIER
%
%  Training of the basic linear classifier where seperation hyperplane
%  is perpedicular to one dimension.
%
%  model = threshold_tr(train_set, sample_weights, labels)
%   train_set: an NxD-matrix, each row is a training sample in the D dimensional feature
%            space.
%        sample_weights: an Nx1-vector, each entry is the weight of the corresponding training sample
%        labels: Nx1 dimensional vector, each entry is the corresponding label (either 1 or 2)
%
%        model: the ouput model. It consists of
%            1) min_error: training error
%            2) min_error_thr: threshold value
%            3) pos_neg: whether up-direction shows the positive region (label:2, 'pos') or
%                the negative region (label:1, 'neg')
%
% Bug Reporting: Please contact the author for bug reporting and comments.
%
% Cuneyt Mertayak
% email: cuneyt.mertayak@gmail.com
% version: 1.0
% date: 21/05/2007

%filter sample
[filtSample,filtLables]=FiltEX(train_set,labels,sample_weights);

%KNN model
% model=fitcknn(filtSample,filtLables);
% model=fitcknn(filtSample,filtLables,'NumNeighbors',5);
% result = predict(model,train_set);
model=svmtrain(filtSample,filtLables);
result = svmclassify(model,train_set);
error_rate = sum(result ~= labels) / length(labels);
[isNoise]=detectNosieWithKNN(filtSample,filtLables,5);
