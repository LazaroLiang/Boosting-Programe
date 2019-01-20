clc;
clear;

load .\data\original_data\Leukemia2.mat
data(:,1)=data(:,1)+1;
data=rot90(data)';
% clear;clc;
% load .\data\original_data\pca_colon.mat
% Sample=rot90(pdata);
% data=Sample';
% id=[16,24,45,51,55,56];
% data(id,:)=[];
% noLableData=data(:,1:end-1);
%  [coeff, score, latent, tsquared, explained] = pca(noLableData);
%  feature_after_PCA=score(:,1:15);
%  data=[feature_after_PCA,data(:,end)];

[m,n]=size(data);
errorCountRecord=zeros(1,m);
weak_learner_n=15;
crossK=5;
iterMax=20;
sum_error=0;
sum1_error=0;
sum_knn=0;
netMaxStep=20;
RightRate = [];
for k=1:iterMax
    indices = crossvalind('Kfold', m, crossK);
    KRight = [];
    for i = 1:crossK %
        test1 = (indices == i);
        train = ~test1;
        trainData = data(train, :);
        testData = data(test1, :);
        TrainOriginalDataX=trainData(:,1:end-1);
        FlattenedData = TrainOriginalDataX(:)'; % 展开矩阵为一列，然后转置为一行。
        MappedFlattened = mapminmax(FlattenedData, 0, 1); % 归一化。
        trainX = reshape(MappedFlattened, size(TrainOriginalDataX)); % 还原为原始矩阵形式。此处不需转置回去，因为reshape恰好是按列重新排序 
        trainY=trainData(:,end);
       
        
        TestOriginalDataX=testData(:,1:end-1);
        FlattenedData = TestOriginalDataX(:)'; % 展开矩阵为一列，然后转置为一行。
        MappedFlattened = mapminmax(FlattenedData, 0, 1); % 归一化。
        testX = reshape(MappedFlattened, size(TestOriginalDataX)); % 还原为原始矩阵形式。此处不需转置回去，因为reshape恰好是按列重新排序
        
        testY=testData(:,end);
        tr_n=length(trainY);
        te_n=length(testY);
        nn=CNNTrain(trainX,trainY);
        [er, bad] = CNNTest(nn, testX, testY);
        KRight = [KRight 1-er];
        
    end
    RightRate = [RightRate mean(KRight)];
%     RightRate
end
MeanRight = mean(RightRate)



% train_x = double(reshape(train_x',28,28,60000))/255;
% test_x = double(reshape(test_x',28,28,10000))/255;
% train_y = double(train_y');
% test_y = double(test_y');
% 
% %% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
% %will run 1 epoch in about 200 second and get around 11% error. 
% %With 100 epochs you'll get around 1.2% error
% 
% rand('state',0)
% 
% cnn.layers = {
%     struct('type', 'i') %input layer
%     struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %sub sampling layer
%     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %subsampling layer
% };
% 
% 
% opts.alpha = 1;
% opts.batchsize = 50;
% opts.numepochs = 1;
% 
% cnn = cnnsetup(cnn, train_x, train_y);
% cnn = cnntrain(cnn, train_x, train_y, opts);
% 
% [er, bad] = cnntest(cnn, test_x, test_y);
% 
% %plot mean squared error
% figure; plot(cnn.rL);
% assert(er<0.12, 'Too big error');