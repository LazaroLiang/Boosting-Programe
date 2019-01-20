close all;
clear;
clc;

% load .\data\original_data\lymphoma.mat
%  load .\data\original_data\nci64.mat
% data=Sample';

load .\data\original_data\Leukemia2.mat
data(:,1)=data(:,1)+1;
data=rot90(data);
noLableData=data(:,1:end-1);
 [coeff, score, latent, tsquared, explained] = pca(noLableData);
 feature_after_PCA=score(:,1:60);
 data=[feature_after_PCA,data(:,end)];

[m,n]=size(data);
classNum=numel(unique(data(:,end)));    %class number
iterators=20;
crossK=5;
sumAdaBoostEveryIter=0;
sumSVMEveryIter=0;
sumRandomForestIter=0;
sumKNNIter=0;
for j=1:iterators
    fprintf('*********The %d th round iterator start***********\n',j);
    indices = crossvalind('Kfold', m, crossK);%??????????3??    
    for i = 1:crossK %??3???????i????????????????????
        test1 = (indices == i);
        train = ~test1;
        trainData = data(train, :);
        testData = data(test1, :);
        trainX=trainData(:,1:end-1);
        trainY=trainData(:,end);
        testX=testData(:,1:end-1);
        testY=testData(:,end);
        if classNum==2
             ada = fitensemble(trainX,trainY,'AdaBoostM1',20,'Discriminant');
        else
            ada = fitensemble(trainX,trainY,'AdaBoostM2',20,'tree');
        end                 
        result = predict(ada,testX);
        AccuracyRate = sum(result == testY) / length(testY);        
        sumAdaBoostEveryIter=sumAdaBoostEveryIter+AccuracyRate;
        fprintf('*********Accuracy: %d ***********\n',AccuracyRate);
    end
    fprintf('*********The %d th round iterator end***********\n',j);
end
fprintf('AdaBoost Average Accuracy:%d\n',sumAdaBoostEveryIter/(iterators*crossK));





% result = DecisionTree(trainX,trainY,testX,testY);   %¾ö²ßÊ÷
% result = NeuroNetwork(trainX,trainY,testX,testY);   %Éñ¾­?øÂç
% result = SVMDecision(trainX,trainY,testX,testY);    %SVM
% result = BayesNaive(trainX,trainY,testX,testY);     %Naive Bayes
% result = AdaBoost(trainX,trainY,testX,testY);       %AdaBoost
% result = RandomForest(trainX,trainY,testX,testY);   %random forest
