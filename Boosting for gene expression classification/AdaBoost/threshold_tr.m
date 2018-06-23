function [model,error_rate,result,filtSample,filtLables,model_name] = threshold_tr(train_set, sample_weights, labels)
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
% knn_model=fitcknn(filtSample,filtLables);%,'NumNeighbors',5
% knn_result = predict(knn_model,train_set);
% 
% svm_model=svmtrain(filtSample,filtLables);
% svm_result = svmclassify(svm_model,train_set);

% dt_model=classregtree(filtSample,filtLables);
% cost =treetest(dt_model,'test',train_set,labels);

% dt_model=fitctree(filtSample,filtLables);
% dt_result=predict(dt_model,train_set);

dt_model=fitcnb(train_set,labels,'Weights' ,sample_weights);
dt_result=predict(dt_model,train_set);
% model
% result = 
% knn_error_rate=sum(knn_result ~= labels) / length(labels);
% svm_error_rate=sum(svm_result ~= labels) / length(labels);

dt_error_rate=sum(dt_result ~= labels) / length(labels);
% if knn_error_rate<svm_error_rate
%     error_rate = knn_error_rate;
%     model=knn_model;
%     result =knn_result;
%     model_name='knn';
% else
%     error_rate = svm_error_rate;
%     model=svm_model;
%     model_name='svm';
%     result =svm_result;
% end
error_rate = dt_error_rate;
    model=dt_model;
    result =dt_result;
    model_name='dt';
    if error_rate>0.5
        disp('error too max')
    end

% [isNoise]=detectNosieWithKNN(filtSample,filtLables,5);
