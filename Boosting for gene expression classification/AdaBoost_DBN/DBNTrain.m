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
%��DBN�ĳ�ʼ��
%���������֮�������㣬ÿ��100����Ԫ����Ϊ�������޲���������
dbn.sizes = [200 200];
%ѵ������
opts.numepochs =   2;
%ÿ���������������
opts.batchsize = size(train_set, 1);

%���·���Ŀǰ��֪����ʲô��
opts.momentum  =   0;
%ѧϰ����
opts.alpha     =   1;
%����DBN
dbn = dbnsetup(dbn, train_set, opts);
%ѵ��DBN
dbn = dbntrain(dbn, train_set, opts);
%���ˣ��������DBN��ѵ��

%unfold dbn to nn
%��DBNѵ���õ�������ת��ΪNN����ʽ
nn = dbnunfoldtonn(dbn, classNums);

%����NN����ֵ����ΪSigmoid����
nn.activation_function = 'sigm';

%train nn
%ѵ��NN
opts.numepochs =  3;
opts.batchsize = size(train_set, 1);
dbnModel = nntrain(nn, train_set, train_label, opts);
%[er, bad] = nntest(nn, test_x, test_y);
%dbnModel = nntrain(nn, train_set, train_label, opts);
end