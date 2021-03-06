close all;
clear;
clc;

% load .\data\original_data\lymphoma.mat
 load .\data\original_data\colon.mat
data=Sample';
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
%         if classNum==2
%              result = AdaBoostM1(trainX,trainY,testX,testY);
%         else
%             result = AdaBoostM2(trainX,trainY,testX,testY);
%         end         
%         sumAdaBoostEveryIter=sumAdaBoostEveryIter+result;
%         
%          if classNum==2
%             result = SVMDecision(trainX,trainY,testX,testY);
%             sumSVMEveryIter=sumSVMEveryIter+result;
%          else
%          end
%         result = BayesNaive(trainX,trainY,testX,testY);
%         result = DecisionTree(trainX,trainY,testX,testY);
%         result = RandomForest(trainX,trainY,testX,testY);
%         sumRandomForestIter=sumRandomForestIter+result;
        
        result = KNN(trainX,trainY,testX,testY);
        sumKNNIter=sumKNNIter+result;
    end
    fprintf('*********The %d th round iterator end***********\n',j);
end
fprintf('AdaBoost Average Accuracy:%d\n',sumAdaBoostEveryIter/(iterators*crossK));
fprintf('SVM Average Accuracy:%d\n',sumSVMEveryIter/(iterators*crossK));
fprintf('RandomForest Average Accuracy:%d\n',sumRandomForestIter/(iterators*crossK));
fprintf('KNN Average Accuracy:%d\n',sumKNNIter/(iterators*crossK));




% result = DecisionTree(trainX,trainY,testX,testY);   %������
% result = NeuroNetwork(trainX,trainY,testX,testY);   %������
% result = SVMDecision(trainX,trainY,testX,testY);    %SVM
% result = BayesNaive(trainX,trainY,testX,testY);     %Naive Bayes
% result = AdaBoost(trainX,trainY,testX,testY);       %AdaBoost
% result = RandomForest(trainX,trainY,testX,testY);   %random forest
