function [diversity]=getDiversity(modelResult,diversityMethod) %���������
[m,n]=size(modelResult);
switch diversityMethod
    case 'Entropy'  %Entropy�����Բ��� 
        temp=n-ceil(n/2);
        sumVar=0;
        for i=1:m
            correctClassfierNum=sum(modelResult(i,:));  %������ȷ�����i�����ķ���������
            minVar=min(correctClassfierNum,n-correctClassfierNum);
            sumVar=sumVar+minVar;
        end
        diversity=sumVar/(temp*m);
    case 'CFD'     %һ����ʧ�ܶ����Բ���
        tempResult=sum(modelResult,2);  %�����з�����Ԥ�������
%         numbers=hist(tempResult, unique(tempResult));
        sumCFD=0;
        for i=1:n   %��ÿ��������
            numFialedByNModel=length(find(tempResult==i));
            numFialedByAtLeastOneModel=length(find(tempResult~=n));
            sumCFD=sumCFD+((n-i)/(n-1))*numFialedByNModel/numFialedByAtLeastOneModel;
        end
        diversity=sumCFD;
end
end