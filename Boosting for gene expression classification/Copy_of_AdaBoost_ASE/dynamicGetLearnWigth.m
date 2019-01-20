%根据待测样本K最近邻的样本在各个模型中的正确率动态设置每个模型的权重
%具体来说找到待测样本的K最近邻样本，然后计算这K个样本在模型下的正确率，根据正确率动态分配每个模型权重
function [probability]=dynamicGetLearnWigth(adaboost_model,te_func_handle,k,test_set)
hypothesis_n = length(adaboost_model.weights);
[m,~]=size(test_set); 
probability=[];     
for i=1:m
    for h=1:hypothesis_n  %对于每个分类器计算K近邻样本中判断正确的概率
        train_set=adaboost_model.train_set{h};
        train_label=adaboost_model.train_lable{h};
        [n,~]=size(train_set);  %获取样本个数，train_set中行为样本数，列为特征向量       
        distance = inf*ones(1,n);
        
        for j=1:n
            distance(i,j) = norm(test_set(i,:)-train_set(j,:));%计算i样本与其它样本之间的距离
        end
        [~,Index]= sort(distance,'ascend');        
        [~,hits,error_rate] = te_func_handle(adaboost_model.parameters{h},train_set(Index(1:k),:),train_label(Index(1:k)), adaboost_model.model_name{h});
        probability(i,h)=hits/k;
    end
end
end