function [saeModel]=SAETrain(train_set,train_label)
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
%     if(train_label(t,1)==2)
%         train_label(t,1)=0;
%         train_label(t,2)=1;
%     else if(train_label(t,1)==3)
%         train_label(t,1)=0;
%         train_label(t,3)=1;
%     end
end
%SAE
        sae = saesetup([n-1 64-1]);
        sae.ae{1}.activation_function       ='sigm';
        sae.ae{1}.learningRate              = 0.5;
%        sae.ae{1}.inputZeroMaskedFraction   = 0;
%         sae.ae{2}.activation_function       ='sigm';
%         sae.ae{2}.learningRate              = 0.5;
%         sae.ae{2}.inputZeroMaskedFraction   = 0;
%          sae.ae{3}.activation_function       ='sigm';
%         sae.ae{3}.learningRate              = 0.5;

        opts.numepochs =   netMaxStep;
%         opts.batchsize = m/crossK;
        opts.batchsize = setSize;
        sae = saetrain(sae, train_set, opts);

        % Use the SDAE to initialize a FFNN
        nn = nnsetup([n-1 64-1 classNums]);
        nn.activation_function              = 'sigm';
        nn.learningRate                     = 0.5;
        nn.W{1} = sae.ae{1}.W{1};
%         nn.W{2} = sae.ae{2}.W{1};
%         nn.W{3} = sae.ae{3}.W{1};

        % Train the FFNN
        opts.numepochs =   netMaxStep;
%         opts.batchsize = m/crossK;
        opts.batchsize = setSize;
        saeModel = nntrain(nn, train_set, train_label, opts);
end