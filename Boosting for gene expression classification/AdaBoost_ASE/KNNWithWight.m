function [test_Predicted_labels] = KNNWithWeight(Xtrain,Ltrain,Weight,varargin)
%% This is the matlab implemenatation for K nearest neighbors
% Input Arguments: 
% Xtrain : training data set
% Ltrain : Labels of training samples
% Xtest : test data set
% K : number of nearest neighbors 
% Output Arguments :
% TestLabel : Predicted labels of the output data set
% default value of K if not given by user is 8.
% if(nargin < 3)
% error('Incorrect number of inputs.');
% end
% if(nargin < 4)
%    K = 8; 
% end

%descendingDistances = zeros(N,Nt);
%Ltest = repmat(Xtest(1,:),N,1);
[FiltXtrain,FiltLtrain]=FiltEX(Xtrain,Ltrain,Weight);

[N , ~] = size(FiltXtrain);
[Ntest,~] = size(Xtrain);
distance = zeros(N,Ntest);

% calculating the euclidean distance of the test samples from training
% samples
for i = 1: Ntest
     for j = 1: N 
        distance(j,i) = norm(Xtrain(i,:)-FiltXtrain(j,:));
     end
end

% ascendingdistances stores all the distances of the test samples
% from the all training samples in cloumns
% Index will have indices of the corresponding training sample
[~,Index]= sort(distance,'ascend');

% consider only top K nearest neighbors to predict the label for test
% sample
Ltest = zeros(K,Ntest);
for i = 1:Ntest
    for j=1:K
    Ltest(j,i) = Ltrain(Index(j,i));
    end
    test_Predicted_labels(i) = mode(Ltest(:,i));
end


test_Predicted_labels = test_Predicted_labels';

end