function [result]=judgeNBIleague(filtSample,filtLables)
%�ж����ر�Ҷ˹��filtSample���Ƿ�����㷨���ԣ����������ڷ���Ϊ0ʱ���Ϸ�
classNum=length(unique(filtLables));
result=1;
for i=1:classNum
    [index]=find(filtLables==i);
%     if var(filtSample(index),1)==0
%         result=0;
%         break;
%     end
    %std(X,0,1)�����������std(X,0,2)�����������
    cube=filtSample(index);
    [m,~]=size(cube);
    flag=-1;
    for j=1:m-1 
        if all(filtSample(index(j),:)==filtSample(index(j+1),:))==0
            flag=1;
            break;
        end
    end
    if flag==-1
        result=0;
        break;
    end
end
end