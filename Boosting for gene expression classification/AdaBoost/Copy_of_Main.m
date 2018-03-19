% clear;clc;
% load .\data\original_data\colon.mat
% data=Sample';
clear;clc;
load .\data\original_data\colon.mat
data=Sample';
id=[16,24,45,51,55,56];
data(id,:)=[];
% noLableData=data(:,1:end-1);
%  [coeff, score, latent, tsquared, explained] = pca(noLableData);
%  feature_after_PCA=score(:,1:15);
%  data=[feature_after_PCA,data(:,end)];

[m,n]=size(data);
errorCountRecord=zeros(1,m);
weak_learner_n=5;
crossK=5;
iterMax=10;
sum_error=0;
sum_knn=0;
for k=1:iterMax
indices = crossvalind('Kfold', m, crossK);
for i = 1:crossK %
        test1 = (indices == i);
        train = ~test1;
        trainData = data(train, :);
        testData = data(test1, :);
        trainX=trainData(:,1:end-1);
        trainY=trainData(:,end);
        testX=testData(:,1:end-1);
        testY=testData(:,end);
        tr_n=length(trainY);
        te_n=length(testY);
        adaboost_model = ADABOOST_tr(@threshold_tr,@threshold_te,trainX,trainY,weak_learner_n);
        % ѵ����������
        [L_tr,hits_tr] = ADABOOST_te(adaboost_model,@threshold_te,trainX,trainY);
        tr_error(i) = (tr_n-hits_tr)/tr_n;
        % ������������
        [L_te,hits_te] = ADABOOST_te(adaboost_model,@threshold_te,testX,testY);
        result=L_te'~=testY;
        for t=1:length(result)
            if result(t) ~= 0
                OriginalIndex=-1;
                for x=1:length(test1)
                    if sum(test1(1:x,:))==t
                        OriginalIndex=x;
                        break;
                    end
                end
                disp(['Original Index:' num2str(OriginalIndex)]);
                errorCountRecord(OriginalIndex)=errorCountRecord(OriginalIndex)+1;
                disp([num2str(t) ' ''s true label is ' num2str(testY(t))]);
            end
        end
        te_error(i) = (te_n-hits_te)/te_n;
        sum_error=sum_error+te_error(i);
        
        knn=fitcknn(trainX,trainY,'NumNeighbors',5);
        resultKNN = predict(knn,testX);
        result=resultKNN~=testY;
        for t=1:length(result)
            if result(t) ~= 0
                 OriginalIndex=-1;
                for x=1:length(test1)
                    if sum(test1(1:x,:))==t
                        OriginalIndex=x;
                        break;
                    end
                end
                disp(['Original Index:' num2str(OriginalIndex)]);
                errorCountRecord(OriginalIndex)=errorCountRecord(OriginalIndex)+1;
                disp([num2str(t) ' ''s true label(KNN) is ' num2str(testY(t))]);
            end
        end
        AccuracyRate = sum(resultKNN == testY) / length(testY);
        sum_knn=sum_knn+AccuracyRate;
%         result = KNN(trainX,trainY,testX,testY);
%         sumKNNIter=sumKNNIter+result;
end
 tr_error
 te_error
end

sum_error/(crossK*iterMax)
sum_knn/(crossK*iterMax)
% mean(te_error)