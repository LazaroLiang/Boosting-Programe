clear;
clc;
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
%对DBN的初始化
%除了输入层之外有两层，每层100个神经元，即为两个受限玻尔兹曼机
dbn.sizes = [100 100];
%训练次数
opts.numepochs =   2;
%每次随机的样本数量
opts.batchsize = 100;
%更新方向，目前不知道有什么用
opts.momentum  =   0;
%学习速率
opts.alpha     =   1;
%建立DBN
dbn = dbnsetup(dbn, train_x, opts);
%训练DBN
dbn = dbntrain(dbn, train_x, opts);
%至此，已完成了DBN的训练

%unfold dbn to nn
%将DBN训练得到的数据转化为NN的形式
nn = dbnunfoldtonn(dbn, 10);

%设置NN的阈值函数为Sigmoid函数
nn.activation_function = 'sigm';

%train nn
%训练NN
opts.numepochs =  3;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');