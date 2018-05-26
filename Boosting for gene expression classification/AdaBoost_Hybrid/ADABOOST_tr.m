function  adaboost_model  =  ADABOOST_tr(tr_func_handle,te_func_handle,train_set,labels,no_of_hypothesis)
%
% ADABOOST TRAINING: A META-LEARNING ALGORITHM
%   adaboost_model = ADABOOST_tr(tr_func_handle,te_func_handle,
%                                train_set,labels,no_of_hypothesis)
%
%         'tr_func_handle' and 'te_func_handle' are function handles for
%         training and testing of a weak learner, respectively. The weak learner
%         has to support the learning in weighted datasets. The prototypes
%         of these functions has to be as follows.
%
%         model = train_func(train_set,sample_weights,labels)
%                     train_set: a TxD-matrix where each row is a training sample in
%                         a D dimensional feature space.
%                     sample_weights: a Tx1 dimensional vector, the i-th entry
%                         of which denotes the weight of the i-th sample.
%                     labels: a Tx1 dimensional vector, the i-th entry of which
%                         is the label of the i-th sample.
%                     model: the output model of the training phase, which can
%                         consists of parameters estimated.
%
%         [L,hits,error_rate] = test_func(model,test_set,sample_weights,true_labels)
%                     model: the output of train_func
%                     test_set: a KxD dimensional matrix, each of whose row is a
%                         testing sample in a D dimensional feature space.
%                     sample_weights:   a Dx1 dimensional vector, the i-th entry
%                         of which denotes the weight of the i-th sample.

%                     true_labels: a Dx1 dimensional vector, the i-th entry of which
%                         is the label of the i-th sample.
%                     L: a Dx1-array with the predicted labels of the samples.
%                     hits: number of hits, calculated with the comparison of L and
%                         true_labels.
%                     error_rate: number of misses divided by the number of samples.
%
%
%         'train_set' contains the samples for training and it is NxD matrix
%         where N is the number of samples and D is the dimension of the
%         feature space. 'labels' is an Nx1 matrix containing the class
%         labels of the samples. 'no_of_hypothesis' is the number of weak
%         learners to be used.
%
%         The output 'adaboost_model' is a structure with the fields
%          - 'weights': 1x'no_of_hypothesis' matrix specifying the weights
%                       of the resulted weighted majority voting combination
%          - 'parameters': 1x'no_of_hypothesis' structure matrix specifying
%                          the special parameters of the hypothesis that is
%                          created at the corresponding iteration of
%                          learning algorithm
%
%         Specific Properties That Must Be Satisfied by The Function pointed
%         by 'func_handle'
%         ------------------------------------------------------------------
%
% Note: Labels must be positive integers from 1 upto the number of classes.
% Node-2: Weighting is done as specified in AIMA book, Stuart Russell et.al. (sec edition)
%
% Bug Reporting: Please contact the author for bug reporting and comments.
%
% Cuneyt Mertayak
% email: cuneyt.mertayak@gmail.com
% version: 1.0
% date: 21/05/2007
%

adaboost_model = struct('weights',zeros(1,no_of_hypothesis),'parameters',[],'train_set',[],'train_lable',[],'model_name',[]); %cell(1,no_of_hypothesis));

sample_n = size(train_set,1);
samples_weight = ones(sample_n,1)/sample_n;
judgeResult=[];%记录每个分类器分类结果，其中，行表示每个样本，列表示分类器对应判定结果，为1表示判定正确，为0表示判定错误
for turn=1:no_of_hypothesis
%     model=tr_func_handle(train_set,samples_weight,labels);
    [model,error_rate,L,filtSample,filtLables,learn_name,tempJudgeResult]=tr_func_handle(train_set,samples_weight,labels,judgeResult);
    adaboost_model.parameters{turn} =model;
    adaboost_model.train_set{turn}=filtSample;
    adaboost_model.train_lable{turn}=filtLables;
    adaboost_model.model_name{turn}=learn_name;
     judgeResult=[judgeResult tempJudgeResult];
%     [L,hits,error_rate]=te_func_handle(adaboost_model.parameters{turn},train_set,samples_weight,labels);
    if(error_rate==1)
        error_rate=1-eps;
    elseif(error_rate==0)
        error_rate=eps;
    end
    rate(turn)=error_rate;
    alpha=1/2*log((1-error_rate)/error_rate);
    % The weight of the turn-th weak classifier
%     adaboost_model.weights(turn) = log10((1-error_rate)/error_rate);
    adaboost_model.weights(turn) = alpha;
%     C=likelihood2class(L);
    t_labeled=(L==labels);  % true labeled samples
%     t_labeled=true(sample_n,1);
% %通过KNN检测样本是否为噪声，train_set为训练样本
%     [isNoise]=detectNosieWithKNN(train_set,labels,7,L);
%     for i=1:sample_n
%         if isNoise(i)==0 && L(i)~=labels(i)  %不是噪声，但是判断错误
%             t_labeled(i)=false; %权重需要加大的样本
%         end
%         if isNoise(i)==1 && L(i)==labels(i)   %是噪声但是判断正确的样本
%             t_labeled(i)=false;
%         end
%     end
    
    % Importance of the true classified samples is decreased for the next weak classifier
    samples_weight(t_labeled) = samples_weight(t_labeled)*exp(-alpha);%判断正确的样本
    samples_weight(~t_labeled) = samples_weight(~t_labeled)*exp(alpha);%判断错误的样本
    
    % Normalization
    samples_weight = samples_weight/sum(samples_weight);
end
rate

% Normalization
adaboost_model.weights=adaboost_model.weights/sum(adaboost_model.weights);