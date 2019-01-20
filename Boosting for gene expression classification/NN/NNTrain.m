function [ nn] = NNTrain( train_x, train_y)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here

[m,n]=size(train_x);
n=n+1;
% n=12601;
netMaxStep=50;
classNums=length(unique(train_y));
setSize=length(train_y);
train_y= train_y(:,1);

train_y = [train_y(:,:) zeros(setSize,classNums-1)];
for t=1:setSize
    l=train_y(t,1);
    if(l~=1)
        train_y(t,1)=0;
        train_y(t,l)=1;
    end
end

% normalize
[train_x, mu, sigma] = zscore(train_x);
% test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([n-1 classNums]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = size(train_x, 1);  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

% [er, bad] = nntest(nn, test_x, test_y);

% assert(er < 0.08, 'Too big error');

end

