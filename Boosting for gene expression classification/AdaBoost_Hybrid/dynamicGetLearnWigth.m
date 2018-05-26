%���ݴ�������K����ڵ������ڸ���ģ���е���ȷ�ʶ�̬����ÿ��ģ�͵�Ȩ��
%������˵�ҵ�����������K�����������Ȼ�������K��������ģ���µ���ȷ�ʣ�������ȷ�ʶ�̬����ÿ��ģ��Ȩ��
function [probability]=dynamicGetLearnWigth(adaboost_model,te_func_handle,k,test_set)
hypothesis_n = length(adaboost_model.weights);
[m,~]=size(test_set); 
probability=[];     
for i=1:m
    for h=1:hypothesis_n  %����ÿ������������K�����������ж���ȷ�ĸ���
        train_set=adaboost_model.train_set{h};
        train_label=adaboost_model.train_lable{h};
        [n,~]=size(train_set);  %��ȡ����������train_set����Ϊ����������Ϊ��������       
        distance = inf*ones(1,n);
        
        for j=1:n
            distance(i,j) = norm(test_set(i,:)-train_set(j,:));%����i��������������֮��ľ���
        end
        [~,Index]= sort(distance,'ascend');        
        [~,hits,error_rate] = te_func_handle(adaboost_model.parameters{h},train_set(Index(1:k),:),train_label(Index(1:k)), adaboost_model.model_name{h});
        probability(i,h)=hits/k;
    end
end
end