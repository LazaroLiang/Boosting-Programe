clear;clc;
load .\data\original_data\colon.mat
dataOriginal=Sample';
filtLableData=dataOriginal(:,1:end-1);
[pc,score,latent,tsquare] = pca(filtLableData);
data=score(:,1:60);
data=[data dataOriginal(:,end)];
[m,n]=size(data);
errorCountRecord=zeros(1,m);
weak_learner_n=20;
crossK=5;
iterMax=20;
sum_error=0;
sum1_error=0;
sum_knn=0;
RightRate=[];
class_num=length(unique(dataOriginal(:,end)));
for k=1:iterMax
indices = crossvalind('Kfold', m, crossK);
KRight = [];
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
        rate(i)=length(find(trainY==1))/tr_n;
        adaboost_model = ADABOOST_tr(@threshold_tr,@threshold_te,trainX,trainY,weak_learner_n);
        % 训练样本测试
%         sprintf('\n')
%         sprintf('*****第%d折训练样本错分情况开始********',i)
        [L_tr,hits_tr] = ADABOOST_te(adaboost_model,@threshold_te,trainX,trainY,class_num);
        tr_error(i) = (tr_n-hits_tr)/tr_n;
%         sprintf('*****第%d折训练样本错分情况结束********',i)
%         sprintf('\n')
%         
%         sprintf('\n')
%         sprintf('*****第%d折测试样本错分情况开始********',i)
        % 测试样本测试
        [L_te,hits_te,another_hits] = ADABOOST_te(adaboost_model,@threshold_te,testX,testY,class_num);
%         sprintf('*****第%d折测试样本错分情况结束********',i)
%         sprintf('\n')
%         result=L_te'~=testY;
%         for t=1:length(result)
%             if result(t) ~= 0
%                 OriginalIndex=-1;
%                 for x=1:length(test1)
%                     if sum(test1(1:x,:))==t
%                         OriginalIndex=x;
%                         break;
%                     end
%                 end
%                 disp(['Original Index:' num2str(OriginalIndex)]);
%                 errorCountRecord(OriginalIndex)=errorCountRecord(OriginalIndex)+1;
%                 disp([num2str(t) ' ''s true label is ' num2str(testY(t))]);
%             end
%         end
        te_error(i) = (te_n-hits_te)/te_n;
        sum_error=sum_error+te_error(i);
        
        another_error(i) = (te_n-another_hits)/te_n;
        sum1_error=sum1_error+another_error(i);
        
        knn=fitcknn(trainX,trainY);%,'NumNeighbors',5
        resultKNN = predict(knn,testX);
%         model=svmtrain(trainX,trainY);
%         resultKNN = svmclassify(model,testX);
        result=resultKNN~=testY;
        AccuracyRate = sum(resultKNN == testY) / length(testY);
        sum_knn=sum_knn+AccuracyRate;
%         model=fitctree(trainX,trainY);
%         resultKNN=predict(model,testX);
%         
%         result=resultKNN~=testY;
        
        nn=SAETrain(trainX,trainY);
%         [er, bad] = SAETest(nn, testX, test_label);
        [er, bad] = SAETest(nn, testX, testY);
        KRight = [KRight 1-er];
         
%         for t=1:length(result)
%             if result(t) ~= 0
%                  OriginalIndex=-1;
%                 for x=1:length(test1)
%                     if sum(test1(1:x,:))==t
%                         OriginalIndex=x;
%                         break;
%                     end
%                 end
%                 disp(['Original Index:' num2str(OriginalIndex)]);
%                 errorCountRecord(OriginalIndex)=errorCountRecord(OriginalIndex)+1;
%                 disp([num2str(t) ' ''s true label(KNN) is ' num2str(testY(t))]);
%             end
%         end
%         AccuracyRate = sum(resultKNN == testY) / length(testY);
%         sum_knn=sum_knn+AccuracyRate;
%         result = KNN(trainX,trainY,testX,testY);
%         sumKNNIter=sumKNNIter+result;
end
tr_error
 te_error
 RightRate = [RightRate mean(KRight)];
end

sum_error/(crossK*iterMax)
% sum1_error/(crossK*iterMax)
1-(sum_knn/(crossK*iterMax))
MeanRight = mean(RightRate);
1-MeanRight
% mean(te_error)