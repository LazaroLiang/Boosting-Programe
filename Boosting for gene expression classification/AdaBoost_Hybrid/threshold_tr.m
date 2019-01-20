function [model,error_rate,result,filtSample,filtLables,model_name,judgeResult,diversity] = threshold_tr(train_set, sample_weights, labels,preJudgeResult,preDiversity)
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
error_rate=1;
while error_rate>0.5
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
if length(preJudgeResult)==0
    dt_model=fitctree(filtSample,filtLables);
    dt_result=predict(dt_model,train_set);
    judgeResult=(dt_result==labels);
    error_rate=sum(dt_result ~= labels) / length(labels);
    model=dt_model;
    result =dt_result;
    model_name='dt';
    diversity=0;
else
    diversityMethod='DFM';
    
    dt_model=fitctree(filtSample,filtLables);
    dt_result=predict(dt_model,train_set);
    dt_temp_result=[preJudgeResult (dt_result==labels)];
    dt_error_rate=sum(dt_result ~= labels) / length(labels);
    dt_diversty=getDiversity(dt_temp_result,diversityMethod,preDiversity);
%     dt_diversty=getDiversity(dt_temp_result,'Entropy');
    

    %% 对于贝叶斯分类器，检查各个类样本方差是否为0，如果为0，则贝叶斯分类器分类时会报错
    if judgeNBIleague(filtSample,filtLables)==0
        filtSample=train_set;
        filtLables=labels;
    end
    nb_model=fitcnb(filtSample,filtLables);
    nb_result=predict(nb_model,train_set);
    nb_temp_result=[preJudgeResult (nb_result==labels)];
    nb_error_rate=sum(nb_result ~= labels) / length(labels);
    nb_diversty=getDiversity(nb_temp_result,diversityMethod,preDiversity);
  % obj = ClassificationDiscriminant.fit(train_data, train_label);  
% predict_label   =       predict(obj, test_data);
% knn_model=fitcdiscr(filtSample,filtLables);
    knn_model=fitcknn(filtSample,filtLables,'NumNeighbors',3);%,'NumNeighbors',5
    knn_result = predict(knn_model,train_set);
    knn_temp_result=[preJudgeResult (knn_result==labels)];
    knn_error_rate=sum(knn_result ~= labels) / length(labels);    
    knn_diversty=getDiversity(knn_temp_result,diversityMethod,preDiversity);
%     knn_diversty=getDiversity(knn_temp_result,'Entropy');
    
%     svm_model=svmtrain(filtSample,filtLables);
%     svm_result = svmclassify(svm_model,train_set);
%     svm_temp_result=[preJudgeResult (svm_result==labels)];
%     svm_error_rate=sum(svm_result ~= labels) / length(labels); 
%     svm_diversty=getDiversity(svm_temp_result,'CFD');
    
%     [~,maxIndex]=max([dt_diversty/dt_error_rate,nb_diversty/nb_error_rate,knn_diversty/knn_error_rate,svm_diversty/svm_error_rate]);
%     maxIndex=randperm(4,1);
%     maxIndex=3;
%     [~,maxIndex]=min([dt_error_rate,nb_error_rate,knn_error_rate]);
    [~,maxIndex]=min([dt_diversty,nb_diversty]);%,knn_diversty
% %     [~,maxIndex]=max([dt_diversty/dt_error_rate,nb_diversty/nb_error_rate]);
%     if maxIndex==2 || maxIndex==4
%         maxIndex=1;
%     end
%     maxIndex=1;
    switch maxIndex
        case 1
            judgeResult=(dt_result==labels);
            error_rate=sum(dt_result ~= labels) / length(labels);
            model=dt_model;
            result =dt_result;
            model_name='dt';
            diversity=dt_diversty;
        case 2
%             dt_diversty
%             knn_diversty
            judgeResult=(nb_result==labels);
            error_rate=sum(nb_result ~= labels) / length(labels);
            model=nb_model;
            result =nb_result;
            model_name='nb';
            diversity=nb_diversty;
        case 3
            judgeResult=(knn_result==labels);
            error_rate=sum(knn_result ~= labels) / length(labels);
            model=knn_model;
            result =knn_result;
            model_name='knn';
            diversity=nb_diversty;
%         case 4
%             judgeResult=(svm_result==labels);
%             error_rate=sum(svm_result ~= labels) / length(labels);
%             model=svm_model;
%             result =svm_result;
%             model_name='svm';
    end
    % model
    % result =
    % knn_error_rate=sum(knn_result ~= labels) / length(labels);
    % svm_error_rate=sum(svm_result ~= labels) / length(labels);
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
end
if error_rate>0.5
    disp('error too max')
end
end

% [isNoise]=detectNosieWithKNN(filtSample,filtLables,5);
