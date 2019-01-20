clear;clc;
load .\data\original_data\nci64.mat

%% LDA 降维
% data=Sample';
% X =Sample(1:end-1 , :);
% Y =Sample(end , :);
%  
%  %%自己加的,用来处理大数据 
%     T = X';
%     Mean_Image = mean(T , 2);
%     T = bsxfun(@minus , T ,Mean_Image);
%     T = T * T'/ (size(unique(Y),2)-1) ;
%     X = T';
%  %%到这里为止feature(:,2:end)
%     
% [Z,W]=FDA(X, Y');
% data = [Z' Y'];
% dataOriginal=Sample';

%% pca降维
dataOriginal=Sample';
filtLableData=dataOriginal(:,1:end-1);
[pc,score,latent,tsquare] = pca(filtLableData);
data=score(:,1:30);
data=[data dataOriginal(:,end)];

% clear;clc;
% load .\data\original_data\pca_colon.mat
% Sample=rot90(pdata);
% 
% % id=[16,24,45,51,55,56];
% % data(id,:)=[];
% noLableData=data(:,1:end-1);
%  [coeff, score, latent, tsquared, explained] = pca(noLableData);
%  feature_after_PCA=score(:,1:60);
%  data=[feature_after_PCA,data(:,end)];

[m,n]=size(data);
errorCountRecord=zeros(1,m);
weak_learner_n=10;
crossK=5;
iterMax=20;
sum_error=0;
sum1_error=0;
sum_knn=0;
sum_dt=0;
sum_nb=0;
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
        rate(i)=length(find(trainY==1))/tr_n;
        adaboost_model = ADABOOST_tr(@threshold_tr,@threshold_te,trainX,trainY,weak_learner_n);
        % 训练样本测试
%         sprintf('\n')
%         sprintf('*****第%d折训练样本错分情况开始********',i)
        [L_tr,hits_tr] = ADABOOST_te(adaboost_model,@threshold_te,trainX,trainY);
        tr_error(i) = (tr_n-hits_tr)/tr_n;
%         sprintf('*****第%d折训练样本错分情况结束********',i)
%         sprintf('\n')
%         
%         sprintf('\n')
%         sprintf('*****第%d折测试样本错分情况开始********',i)
        % 测试样本测试
        [L_te,hits_te,another_hits] = ADABOOST_te(adaboost_model,@threshold_te,testX,testY);
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
        result=resultKNN~=testY;
%         model=svmtrain(trainX,trainY);
%         resultKNN = svmclassify(model,testX);
        
        dt=fitctree(trainX,trainY);
        resultDT=predict(dt,testX);
        
        nb=fitctree(trainX,trainY);
        resultNB=predict(nb,testX);
        
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
        AccuracyRate = sum(resultKNN == testY) / length(testY);
        sum_knn=sum_knn+AccuracyRate;
        
        sum_dt=sum_dt+sum(resultDT == testY) / length(testY);
        sum_nb=sum_nb+sum(resultNB == testY) / length(testY);
        
       adaboost_model.model_name
%         result = KNN(trainX,trainY,testX,testY);
%         sumKNNIter=sumKNNIter+result;
end
tr_error
 te_error
 %rate
end

disp(['ensamble error rate:',num2str(sum_error/(crossK*iterMax))]);
%sum1_error/(crossK*iterMax)
disp(['knn error rate:',num2str(1-(sum_knn/(crossK*iterMax)))]);
disp(['decsion tree error rate:',num2str(1-(sum_dt/(crossK*iterMax)))]); 
disp(['navie bayes error rate:',num2str(1-(sum_nb/(crossK*iterMax)))]); 

% mean(te_error)