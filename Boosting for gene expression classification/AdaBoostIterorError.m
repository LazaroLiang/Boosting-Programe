close all;
clear;
clc;

% load .\data\original_data\lymphoma.mat
 load .\data\original_data\colon.mat  %nci64.mat
data=Sample';
[m,n]=size(data);

%Discriminant
train_data=data(:,1:end-1);
label_data=data(:,end);
% ada = fitensemble(train_data,label_data,'AdaBoostM2',300,'tree');%,'Holdout',1
%kfoldLoss(ada,'mode','cumulative')
%  plot(kfoldLoss(ada,'mode','cumulative'));
% xlabel('Number of decision trees');
% ylabel('Holdout error');
crossK=5;
accuracy=[];
classfierNum=200;
for j=1:1:classfierNum
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
        ada = fitensemble(trainX,trainY,'AdaBoostM1',j,'Discriminant');               
        result = predict(ada,testX);
        AccuracyRate = sum(result == testY) / length(testY);        
        sumAdaBoostEveryIter=sumAdaBoostEveryIter+AccuracyRate;      
    end
   fprintf('*********Step iterators:%d    Average Accuracy: %d ***********\n',j,sumAdaBoostEveryIter/crossK);
   accuracy=[accuracy,sumAdaBoostEveryIter/crossK];
end

plot(1:1:classfierNum,accuracy);

% classNum=numel(unique(data(:,end)));    %class number
% iterators=20;
% crossK=5;
% sumAdaBoostEveryIter=0;
% sumSVMEveryIter=0;
% sumRandomForestIter=0;
% sumKNNIter=0;
% for j=1:iterators
%     fprintf('*********The %d th round iterator start***********\n',j);
%     indices = crossvalind('Kfold', m, crossK);%??????????3??    
%     for i = 1:crossK %??3???????i????????????????????
%         test1 = (indices == i);
%         train = ~test1;
%         trainData = data(train, :);
%         testData = data(test1, :);
%         trainX=trainData(:,1:end-1);
%         trainY=trainData(:,end);
%         testX=testData(:,1:end-1);
%         testY=testData(:,end);
%         if classNum==2
%              ada = fitensemble(trainX,trainY,'AdaBoostM1',200,'Discriminant');
%         else
%             ada = fitensemble(trainX,trainY,'AdaBoostM2',200,'tree');
%         end                 
%         result = predict(ada,testX);
%         AccuracyRate = sum(result == testY) / length(testY);        
%         sumAdaBoostEveryIter=sumAdaBoostEveryIter+AccuracyRate;
%         fprintf('*********Accuracy: %d ***********\n',AccuracyRate);
%     end
%     fprintf('*********The %d th round iterator end***********\n',j);
% end
% fprintf('AdaBoost Average Accuracy:%d\n',sumAdaBoostEveryIter/(iterators*crossK));
