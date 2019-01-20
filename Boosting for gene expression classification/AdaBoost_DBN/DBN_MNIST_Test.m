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
%��DBN�ĳ�ʼ��
%���������֮�������㣬ÿ��100����Ԫ����Ϊ�������޲���������
dbn.sizes = [100 100];
%ѵ������
opts.numepochs =   2;
%ÿ���������������
opts.batchsize = 100;
%���·���Ŀǰ��֪����ʲô��
opts.momentum  =   0;
%ѧϰ����
opts.alpha     =   1;
%����DBN
dbn = dbnsetup(dbn, train_x, opts);
%ѵ��DBN
dbn = dbntrain(dbn, train_x, opts);
%���ˣ��������DBN��ѵ��

%unfold dbn to nn
%��DBNѵ���õ�������ת��ΪNN����ʽ
nn = dbnunfoldtonn(dbn, 10);

%����NN����ֵ����ΪSigmoid����
nn.activation_function = 'sigm';

%train nn
%ѵ��NN
opts.numepochs =  3;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');