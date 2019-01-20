clc;
clear all;
close all;

ctime=datestr(now,30);
tseed = str2num(ctime((end-5):end));
rand('seed',tseed);
tic;
%% ���ݼ��غͼ��
%62*4026  3��
load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\original\lymphoma.mat');
Sample=Sample';
data = [Sample(:,end) Sample(:,1:end-1)];
%�����ݽ��б�׼��
% data = zscore(Sample(:,1:end-1));
% data = [Sample(:,end) data];

% %���� PCA+FDA ��ά�������
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\pf\lymphoma_pfdata_big.mat');

%���� PCA ��ά�������
% load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\pca\pca_lymphoma.mat');
% data = [feature(:,1) zscore(feature(:,2:end))];

%���� NMF ��ά�������
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\nmf\nmf_lymphoma.mat');
%data = [ndata(:,1) zscore(ndata(:,2:end))];

%���� FDA ��ά�������
% load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\fda\lymphoma.mat');
% data = [fdata(:,1) zscore(fdata(:,2:end))];

%%���� NMF+FDA ��ά�������
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\nmf\nmf_lymphoma.mat');
%load('D:\R2015b_win64\2015b\toolbox\DeepLearnToolbox-master\data\fda\lymphoma.mat');
%data = [ndata(:,1) zscore(ndata(:,2:end)) zscore(fdata(:,2:end))];

%�����5�ؽ�����֤���������һ��
data = [data;data(1:3,:)];
%data�����Ǵ��������������Ϊ�����ҵ�һ�ж�Ϊ��ǩ
[ROW,COL]=size(data);
%% �����ݽ���Ԥ����
% for j = 2:COL
%     [maxdata,index]=max(data(:,j));
%     [mindata,index]=min(data(:,j));
%     data(:,j)= (data(:,j)-mindata)./(maxdata - mindata);
% end
%% �õ����ݱ�ǩ
data_label= data(:,1);
data_label = [data_label(:,:) zeros(ROW,2)];
%��data_label����ɾ�����ʽ
for i=1:ROW
   if(data_label(i,1)==2)
       data_label(i,1)=0;
       data_label(i,2)=1;
   elseif(data_label(i,1)==3)
       data_label(i,1)=0;
       data_label(i,3)=1;
    end   
end  
%% ����20��
maxStep = 5;
%���繹��Ĵ���
netMaxStep=15;
RightRate = [];
for step=1:maxStep
    % ����5�۽���
    numOfKfold = 5;
    randRow = randperm(ROW);
    %indices = crossvalind('Kfold',data(1:M,N),numOfKfold);
    KRight = [];
    for k= 1:numOfKfold
        %��������
        test_data = data(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW),2:end);
        test_label = data_label(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW),:);
        %ѵ������
        train_data = data(:,2:end);
        train_data(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW ),:)=[];
        train_label = data_label(:,:);
        train_label(randRow((k-1)/numOfKfold*ROW+1 : k/numOfKfold*ROW ),:)=[];
        %SAE
        sae = saesetup([COL-1 64-1]);
        sae.ae{1}.activation_function       ='sigm';
        sae.ae{1}.learningRate              = 0.5;
       % sae.ae{1}.inputZeroMaskedFraction   = 0;

        opts.numepochs =   netMaxStep;
        opts.batchsize = ROW/numOfKfold;
        sae = saetrain(sae, train_data, opts);

        % Use the SDAE to initialize a FFNN
        nn = nnsetup([COL-1 64-1 3]);
        nn.activation_function              = 'sigm';
        nn.learningRate                     = 0.5;
        nn.W{1} = sae.ae{1}.W{1};

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
%% ������ȷ�ʵ�ͼ
plot(RightRate,'b-+','LineWidth',2);
xlabel('��������');
ylabel('������ȷ��');
axis([0 maxStep 0 1]);
%% ����ά����������浽�ļ���
 t=nnff(nn,data(:,2:end),data_label);
 %t.a{end-1}�ĵ�һ�������ӵ�ƫ��ֵ
 deepFeature=t.a{end-1};
 
 %  %2�ؽ���
%  sample=[data(:,1) sample(:,2:end)];

%5�ؽ���
 deepFeature=[data(1:end-3,1) deepFeature(1:end-3,2:end)];
 
 %save D:\ProgramFiles\MATLAB\R2011b\toolbox\DeepLearnToolbox-master\data\deeplearning\deep_lymphoma.mat  deepFeature;
 toc
  %endtime = datestr(now,13)