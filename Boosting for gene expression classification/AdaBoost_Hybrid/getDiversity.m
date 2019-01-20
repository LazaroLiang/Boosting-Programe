function [diversity]=getDiversity(modelResult,diversityMethod,preDiversity) %���������
%modelResult:���з�������Ԥ����������ÿ�д���һ���������д����Ӧ��������������1��ʾ������ȷ��0��ʾ�������
%diversityMethod����ʾ�����Զ�������
%preDiversity����һ�ֶ�����ֵ
[m,n]=size(modelResult);
switch diversityMethod
    case 'Entropy'  %Entropy�����Բ��� 
        temp=n-ceil(n/2);
        sumVar=0;
        for i=1:m
            correctClassfierNum=sum(modelResult(i,:));  %������ȷ�����i�����ķ���������
            miVnar=min(correctClassfierNum,n-correctClassfierNum);
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
    case 'DFM'  %˫��������ɶԶ������ԣ�ͨ������ÿһ�Ի������������ԣ�Ȼ�����ƽ��ֵ
        pairDiversitySum=0;
        for i=1:n-1
            tempResult=modelResult(i,:)+modelResult(i,n);
            pairDiversitySum=pairDiversitySum+sum(tempResult()==0)/m;
        end
        if preDiversity==0
            diversity=pairDiversitySum/(n-1);
        else
            diversity=(pairDiversitySum/(n-1)+preDiversity)/2;
        end
end
end