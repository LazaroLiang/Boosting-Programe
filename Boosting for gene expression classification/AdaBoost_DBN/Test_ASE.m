clear;clc;
tic;
load .\data\original_data\lymphoma.mat
data=Sample';
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
iterMax=5;
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
        trainX=trainData(:,1:end-1);
        trainY=trainData(:,end);
        testX=testData(:,1:end-1);
        testY=testData(:,end);
        tr_n=length(trainY);
        te_n=length(testY);
        
%         train_label= trainY(:,1);
%         train_label = [train_label(:,:) zeros(tr_n,1)];
%         for t=1:tr_n
%             if(train_label(t,1)==2)
%                 train_label(t,1)=0;
%                 train_label(t,2)=1;
%             end
%         end
%         test_label= testY(:,1);
%         test_label = [test_label(:,:) zeros(te_n,1)];
%         for t=1:te_n
%             if(test_label(t,1)==2)
%                 test_label(t,1)=0;
%                 test_label(t,2)=1;
%             end
%         end
%         
%         %SAE
%         sae = saesetup([n-1 63-1]);
%         sae.ae{1}.activation_function       ='sigm';
%         sae.ae{1}.learningRate              = 0.5;
%        % sae.ae{1}.inputZeroMaskedFraction   = 0;
% %         sae.ae{2}.activation_function       ='sigm';
% %         sae.ae{2}.learningRate              = 0.5;
% %          sae.ae{3}.activation_function       ='sigm';
% %         sae.ae{3}.learningRate              = 0.5;
% 
%         opts.numepochs =   netMaxStep;
% %         opts.batchsize = m/crossK;
%         opts.batchsize = tr_n;
%         sae = saetrain(sae, trainX, opts);
% 
%         % Use the SDAE to initialize a FFNN
%         nn = nnsetup([n-1 63-1 2]);
%         nn.activation_function              = 'sigm';
%         nn.learningRate                     = 0.5;
%         nn.W{1} = sae.ae{1}.W{1};
% %         nn.W{2} = sae.ae{2}.W{1};
% %         nn.W{3} = sae.ae{3}.W{1};
% 
%         % Train the FFNN
%         opts.numepochs =   netMaxStep;
% %         opts.batchsize = m/crossK;
%         opts.batchsize = tr_n;
%         nn = nntrain(nn, trainX, train_label, opts);
        nn=SAETrain(trainX,trainY);
%         [er, bad] = SAETest(nn, testX, test_label);
        [er, bad] = SAETest(nn, testX, testY);
        KRight = [KRight 1-er];
        
    end
    RightRate = [RightRate mean(KRight)];
    RightRate
end
MeanRight = mean(RightRate)
%% 画出正确率的图
plot(RightRate,'b-+','LineWidth',2);
xlabel('迭代次数');
ylabel('分类正确率');
axis([0 iterMax 0 1]);
%% 将降维后的特征保存到文件中
%  t=nnff(nn,data(:,2:end),data_label);
%  %t.a{end-1}的第一列是增加的偏置值
%  deepFeature=t.a{end-1};
 
 %  %2重交叉
%  sample=[data(:,1) sample(:,2:end)];

%5重交叉
%  deepFeature=[data(1:end-3,1) deepFeature(1:end-3,2:end)];
 %save D:\ProgramFiles\MATLAB\R2011b\toolbox\DeepLearnToolbox-master\data\deeplearning\deep_colon.mat  deepFeature;
 toc
