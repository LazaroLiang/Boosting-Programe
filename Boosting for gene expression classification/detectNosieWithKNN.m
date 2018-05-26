%通过KNN检测样本是否为噪声
%%
%train_set：表示训练样本，其中行表示样本，列表示特征
%true_label:表示训练样本的真实类标
%K：表示KNN中K的取值
%predict_label:表示采用每一轮训练出来的模型对训练样本进行预测的类标值
%%

function [isNoise]=detectNosieWithKNN(train_set,true_label,k,predict_label)
    [n,~]=size(train_set);  %获取样本个数，train_set中行为样本数，列为特征向量
    distance = inf*ones(n,n);
    isNoise=zeros(1,n);
    noisePosibale=[];
    for i=1:n
        for j=1:n
            if i~=j
                distance(j,i) = norm(train_set(i,:)-train_set(j,:));%计算i样本与其它样本之间的距离
            end
        end
        [~,Index]= sort(distance,'ascend');
        count=0;
        for j=1:k
           if true_label(Index(j,i))~=predict_label(Index(j,i))
                count=count+1;
           end
        end
        noisePosibale(i)=count/k;   %计算K个近邻中预测错误的概率
    end
    for i=1:n
        if noisePosibale(i)>=3/k %mean(noisePosibale)*1.2 %如果i样本K个近邻预测错误的概率大于所有样本预测的平均值，则认为i样本为噪声
            isNoise(i)=1;
        end
    end
    
end