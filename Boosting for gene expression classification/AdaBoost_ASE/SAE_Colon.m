clc;
clear all;
close all;

ctime=datestr(now,30);
tseed = str2num(ctime((end-5):end));
rand('seed',tseed);

tic;
%% 数据加载和检查
%colon 62*2000 2类
load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\original\colon.mat');
Sample=Sample';
%data = [Sample(:,end) Sample(:,1:end-1)];

%对数据进行归一化
%[data , PS] = mapminmax(Sample(: , 1:end-1) , 0 , 1);
%对数据进行标准化
data = zscore(Sample(:,1:end-1));
data = [Sample(:,end) data];

%加载 PCA+FDA 降维后的数据
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\pf\colon_pfdata.mat');

%加载 PCA 降维后的数据
% load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\pca\pca_colon.mat');
% data = [pdata(:,1) zscore(pdata(:,2:end))];

%加载 NMF 降维后的数据
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\nmf\nmf_colon.mat');
%data = [ndata(:,1) zscore(ndata(:,2:end))];

%加载 FDA 降维后的数据
% load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\fda\colon.mat');
% data = [fdata(:,1) zscore(fdata(:,2:end))];

%%加载 NMF+FDA 降维后的数据
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\nmf\nmf_colon.mat');
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\fda\colon.mat');
%data = [ndata(:,1) zscore(ndata(:,2:end)) zscore(fdata(:,2:end))];


%如果采用5重交叉，则加上这句话
data = [data;data(1:3,:)];

%data的行是待分类的样本，列为特征且第一列都为标签
[ROW,COL]=size(data);

%% 得到数据标签
data_label= data(:,1);
data_label = [data_label(:,:) zeros(ROW,1)];
%将data_label处理成矩阵形式
for i=1:ROW
   if(data_label(i,1)==2)
       data_label(i,1)=0;
       data_label(i,2)=1;
    end   
end  
%% 迭代20次
maxStep = 10;
%网络构造的次数
netMaxStep=20;
RightRate = [];
for step=1:maxStep
    % 采用5重交叉验证
    numOfKfold = 5;
    randRow = randperm(ROW);
    %indices = crossvalind('Kfold',data(1:M,N),numOfKfold);
    KRight = [];
    for k= 1:numOfKfold
        %测试数据
        test_data = data(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW),2:end);
        test_label = data_label(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW),:);
        %训练数据
        train_data = data(:,2:end);
        train_data(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW ),:)=[];
        train_label = data_label(:,:);
        train_label(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW ),:)=[];
        %SAE
        sae = saesetup([COL-1 63-1]);
        sae.ae{1}.activation_function       ='sigm';
        sae.ae{1}.learningRate              = 0.5;
       % sae.ae{1}.inputZeroMaskedFraction   = 0;
%         sae.ae{2}.activation_function       ='sigm';
%         sae.ae{2}.learningRate              = 0.5;
%          sae.ae{3}.activation_function       ='sigm';
%         sae.ae{3}.learningRate              = 0.5;

        opts.numepochs =   netMaxStep;
        opts.batchsize = ROW/numOfKfold;
        sae = saetrain(sae, train_data, opts);

        % Use the SDAE to initialize a FFNN
        nn = nnsetup([COL-1 63-1 2]);
        nn.activation_function              = 'sigm';
        nn.learningRate                     = 0.5;
        nn.W{1} = sae.ae{1}.W{1};
%         nn.W{2} = sae.ae{2}.W{1};
%         nn.W{3} = sae.ae{3}.W{1};

        % Train the FFNN
        opts.numepochs =   netMaxStep;
        opts.batchsize = ROW/numOfKfold;
        nn = nntrain(nn, train_data, train_label, opts);
    
        [er, bad] = nntest(nn, test_data, test_label);
        KRight = [KRight 1-er];
    end
    RightRate = [RightRate mean(KRight)];
end

MeanRight = mean(RightRate)
%% 画出正确率的图
plot(RightRate,'b-+','LineWidth',2);
xlabel('迭代次数');
ylabel('分类正确率');
axis([0 maxStep 0 1]);
%% 将降维后的特征保存到文件中
 t=nnff(nn,data(:,2:end),data_label);
 %t.a{end-1}的第一列是增加的偏置值
 deepFeature=t.a{end-1};
 
 %  %2重交叉
%  sample=[data(:,1) sample(:,2:end)];

%5重交叉
 deepFeature=[data(1:end-3,1) deepFeature(1:end-3,2:end)];
 %save D:\ProgramFiles\MATLAB\R2011b\toolbox\DeepLearnToolbox-master\data\deeplearning\deep_colon.mat  deepFeature;
 toc
% endtime = datestr(now,13)