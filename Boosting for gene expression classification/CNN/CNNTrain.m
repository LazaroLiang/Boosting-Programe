function [cnnModel]=CNNTrain(train_set,train_label)
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
rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

cnn = cnnsetup(cnn, train_set, train_label);
cnnModel = cnntrain(cnn, train_set, train_label, opts);

% [er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
% figure; plot(cnn.rL);
% assert(er<0.12, 'Too big error');
end