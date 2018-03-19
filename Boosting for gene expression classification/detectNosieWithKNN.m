%通过KNN检测样本是否为噪声
function [isNoise]=detectNosieWithKNN(train_set,labels,k)
    [n,~]=size(train_set);  %获取样本个数，train_set中行为样本数，列为特征向量
    distance = zeros(n,n);
    for i=1:n
        for j=1:n
            if i~=j
                distance(j,i) = norm(train_set(i,:)-train_set(j,:));%计算i样本与其它样本之间的距离
            end
        end
        [~,Index]= sort(distance,'ascend');
        for j=1:K
            Ltest(j,i) = Ltrain(Index(j,i));
        end
    end
end