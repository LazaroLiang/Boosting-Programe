clear;clc;
tic;
load .\data\original_data\colon.mat
data=Sample';

% clear;clc;
% load .\data\original_data\Brain_Tumor1.mat
% data(:,1)=data(:,1)+1;
% data=rot90(data)';

% clear;clc;
% load .\data\original_data\pca_colon.mat
% Sample=rot90(pdata);
% data=Sample';
% id=[16,24,45,51,55,56];
% data(id,:)=[];
noLableData=data(:,1:end-1);
 [coeff, score, latent, tsquared, explained] = pca(noLableData);
 feature_after_PCA=score(:,1:60);
 data=[feature_after_PCA,data(:,end)];

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
        FlattenedData = TrainOriginalDataX(:)'; % չ������Ϊһ�У�Ȼ��ת��Ϊһ�С�
        MappedFlattened = mapminmax(FlattenedData, 0, 1); % ��һ����
        trainX = reshape(MappedFlattened, size(TrainOriginalDataX)); % ��ԭΪԭʼ������ʽ���˴�����ת�û�ȥ����Ϊreshapeǡ���ǰ����������� 
        trainY=trainData(:,end);
       
        
        TestOriginalDataX=testData(:,1:end-1);
        FlattenedData = TestOriginalDataX(:)'; % չ������Ϊһ�У�Ȼ��ת��Ϊһ�С�
        MappedFlattened = mapminmax(FlattenedData, 0, 1); % ��һ����
        testX = reshape(MappedFlattened, size(TestOriginalDataX)); % ��ԭΪԭʼ������ʽ���˴�����ת�û�ȥ����Ϊreshapeǡ���ǰ�����������
        
        testY=testData(:,end);
        tr_n=length(trainY);
        te_n=length(testY);
%         nn=NNTrain(trainX,trainY);
%         [er, bad] = NNTest(nn, testX, testY);
%         KRight = [KRight 1-er];
        
%         nn=NNTrain(TrainOriginalDataX,trainY);
%         [er, bad] = NNTest(nn, TestOriginalDataX, testY);
%         KRight = [KRight 1-er];
        KRight=NeuroNetwork(TrainOriginalDataX,trainY,TestOriginalDataX,testY);
%         NeuroNetwork
        
    end
    RightRate = [RightRate mean(KRight)]
%     RightRate
end
MeanRight = mean(RightRate)