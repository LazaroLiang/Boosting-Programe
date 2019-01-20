function [dbnModel]=DBNTrain(train_set,train_label)
%n=2001;
[m,n]=size(train_set);
n=n+1;
% n=12601;
netMaxStep=50;
classNums=length(unique(train_label));
setSize=length(train_label);
train_label= train_label(:,1);

train_label = [train_label(:,:) zeros(setSize,classNums-1)];
for t=1:setSize
    l=train_label(t,1);
    if(l~=1)
        train_label(t,1)=0;
        train_label(t,l)=1;
    end
end
%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
%对DBN的初始化
%除了输入层之外有两层，每层100个神经元，即为两个受限玻尔兹曼机
dbn.sizes = [200 200];
%训练次数
opts.numepochs =   2;
%每次随机的样本数量
opts.batchsize = size(train_set, 1);

%更新方向，目前不知道有什么用
opts.momentum  =   0;
%学习速率
opts.alpha     =   1;
%建立DBN
dbn = dbnsetup(dbn, train_set, opts);
%训练DBN
dbn = dbntrain(dbn, train_set, opts);
%至此，已完成了DBN的训练

%unfold dbn to nn
%将DBN训练得到的数据转化为NN的形式
nn = dbnunfoldtonn(dbn, classNums);

%设置NN的阈值函数为Sigmoid函数
nn.activation_function = 'sigm';

%train nn
%训练NN
opts.numepochs =  3;
opts.batchsize = size(train_set, 1);
dbnModel = nntrain(nn, train_set, train_label, opts);
%[er, bad] = nntest(nn, test_x, test_y);
%dbnModel = nntrain(nn, train_set, train_label, opts);
end