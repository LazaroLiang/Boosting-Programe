%ͨ��KNN��������Ƿ�Ϊ����
function [isNoise]=detectNosieWithKNN(train_set,labels,k)
    [n,~]=size(train_set);  %��ȡ����������train_set����Ϊ����������Ϊ��������
    distance = zeros(n,n);
    for i=1:n
        for j=1:n
            if i~=j
                distance(j,i) = norm(train_set(i,:)-train_set(j,:));%����i��������������֮��ľ���
            end
        end
        [~,Index]= sort(distance,'ascend');
        for j=1:K
            Ltest(j,i) = Ltrain(Index(j,i));
        end
    end
end