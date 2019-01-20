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
% model
% result = 
% knn_error_rate=sum(knn_result ~= labels) / length(labels);
% svm_error_rate=sum(svm_result ~= labels) / length(labels);

% dt_error_rate=sum(dt_result ~= labels) / length(labels);
%filter sample
error_rate=1;
times=0;
while error_rate>0.6
%         disp('error too max')
        [filtSample,filtLables]=FiltEX(train_set,labels,sample_weights);
        while length(unique(filtLables))<length(unique(labels))
            disp('filt data set class number is small then all classes')
            [filtSample,filtLables]=FiltEX(train_set,labels,sample_weights);
        end
        
%         if(length(unique(filtLables))<=2)
%             disp(['class number is smaller then all calsses:' ]);
%             disp(unique(filtLables));
%         end
        sae_model=SAETrain(filtSample,filtLables);
        sae_result=SAEPredict(sae_model,train_set);
        sae_error_rate=sum(sae_result~=labels)/ length(labels);
        error_rate = sae_error_rate;
        model=sae_model;
        result =sae_result;
        model_name='sae';
        times=times+1;
end
if times>=2
    disp(['error too max times:' num2str(times)])
end
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
% error_rate = dt_error_rate;
%     model=dt_model;
%     result =dt_result;
%     model_name='dt';
%     if error_rate>0.4
%         disp('error too max')
%     end

% [isNoise]=detectNosieWithKNN(filtSample,filtLables,5);
