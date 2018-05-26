%ͨ��KNN��������Ƿ�Ϊ����
%%
%train_set����ʾѵ�������������б�ʾ�������б�ʾ����
%true_label:��ʾѵ����������ʵ���
%K����ʾKNN��K��ȡֵ
%predict_label:��ʾ����ÿһ��ѵ��������ģ�Ͷ�ѵ����������Ԥ������ֵ
%%

function [isNoise]=detectNosieWithKNN(train_set,true_label,k,predict_label)
    [n,~]=size(train_set);  %��ȡ����������train_set����Ϊ����������Ϊ��������
    distance = inf*ones(n,n);
    isNoise=zeros(1,n);
    noisePosibale=[];
    for i=1:n
        for j=1:n
            if i~=j
                distance(j,i) = norm(train_set(i,:)-train_set(j,:));%����i��������������֮��ľ���
            end
        end
        [~,Index]= sort(distance,'ascend');
        count=0;
        for j=1:k
           if true_label(Index(j,i))~=predict_label(Index(j,i))
                count=count+1;
           end
        end
        noisePosibale(i)=count/k;   %����K��������Ԥ�����ĸ���
    end
    for i=1:n
        if noisePosibale(i)>=3/k %mean(noisePosibale)*1.2 %���i����K������Ԥ�����ĸ��ʴ�����������Ԥ���ƽ��ֵ������Ϊi����Ϊ����
            isNoise(i)=1;
        end
    end
    
end