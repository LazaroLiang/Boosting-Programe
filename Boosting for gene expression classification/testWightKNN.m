% Xtrain=[5,1;6,3;7,2;3,2;4,5]
% Ltrain=[1,1,1,2,2]
% Xtest=[3,3;6,5]
% K=3;
% Wight=[0.2,0.2,0.2,0.2,0.2];
clc;
clear;
 load .\data\original_data\nci64.mat  %colon.mat
data=Sample';
[m,n]=size(data);
crossK=3;
K=1;
indices = crossvalind('Kfold', m, crossK);%??????????3??    
    sumAdaBoostEveryIter=0;
    for i = 1:crossK %??3???????i????????????????????
        test1 = (indices == i);
        train = ~test1;
        trainData = data(train, :);
        testData = data(test1, :);
        trainX=trainData(:,1:end-1);
        trainY=trainData(:,end);
        testX=testData(:,1:end-1);
        testY=testData(:,end);
        len=length(trainY);
        Wight=1/len*ones(len,1);
        result=KNNWithWight(trainX,trainY,testX, K,Wight);
%         ada = fitensemble(trainX,trainY,'AdaBoostM1',j,'Discriminant');               
%         result = predict(ada,testX);
        AccuracyRate = sum(result == testY) / length(testY)        
        sumAdaBoostEveryIter=sumAdaBoostEveryIter+AccuracyRate;      
    end