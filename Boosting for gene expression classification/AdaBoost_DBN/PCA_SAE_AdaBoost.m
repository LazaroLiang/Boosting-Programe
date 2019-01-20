function [ensamble_error]=PCA_SAE_AdaBoost(sample,dimension,weak_learner_n,crossK,iterMax)
dataOriginal=sample';
filtLableData=dataOriginal(:,1:end-1);
[pc,score,latent,tsquare] = pca(filtLableData);
data=score(:,1:dimension);
data=[data dataOriginal(:,end)];


[m,n]=size(data);
errorCountRecord=zeros(1,m);
%weak_learner_n=10;
%crossK=5;
%iterMax=2;
sum_error=0;
sum1_error=0;
sum_knn=0;
sumSVM=0;
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
        % ÑµÁ·Ñù±¾Îó²î
        [L_tr,hits_tr] = ADABOOST_te(adaboost_model,@threshold_te,trainX,trainY,class_num);
        tr_error(i) = (tr_n-hits_tr)/tr_n;

        % ²âÊÔÑù±¾Îó²î
        [L_te,hits_te,another_hits] = ADABOOST_te(adaboost_model,@threshold_te,testX,testY,class_num);
        te_error(i) = (te_n-hits_te)/te_n;
        sum_error=sum_error+te_error(i);
        
        another_error(i) = (te_n-another_hits)/te_n;
        sum1_error=sum1_error+another_error(i);
        
        knn=fitcknn(trainX,trainY);%,'NumNeighbors',5
        resultKNN = predict(knn,testX);

        model=svmtrain(trainY,trainX,'-t 1');%,'-s 1 ,'-t 0''
        [svmPredictLable]=svmpredict(testY,testX,model);
        resultSVM = sum(svmPredictLable == testY) / length(testY);
        sumSVM=sumSVM+resultSVM;
        
        result=resultKNN~=testY;
        AccuracyRate = sum(resultKNN == testY) / length(testY);
        sum_knn=sum_knn+AccuracyRate;
        
        nn=SAETrain(trainX,trainY);
        [er, bad] = SAETest(nn, testX, testY);
        KRight = [KRight 1-er];
end
tr_error
 te_error
 RightRate = [RightRate mean(KRight)];
end
disp(['ensamble error rate:',num2str(sum_error/(crossK*iterMax))]);
disp(['svm error rate:',num2str(1-(sumSVM/(crossK*iterMax)))]);
disp(['knn error rate:',num2str(1-(sum_knn/(crossK*iterMax)))]);
disp(['sae error rate:',num2str(1-mean(RightRate))]); 
ensamble_error=sum_error/(crossK*iterMax);
end