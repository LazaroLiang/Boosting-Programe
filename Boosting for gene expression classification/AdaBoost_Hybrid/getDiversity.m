function [diversity]=getDiversity(modelResult,diversityMethod) %计算多样性
[m,n]=size(modelResult);
switch diversityMethod
    case 'Entropy'  %Entropy多样性策略 
        temp=n-ceil(n/2);
        sumVar=0;
        for i=1:m
            correctClassfierNum=sum(modelResult(i,:));  %计算正确分类的i样本的分类器个数
            minVar=min(correctClassfierNum,n-correctClassfierNum);
            sumVar=sumVar+minVar;
        end
        diversity=sumVar/(temp*m);
    case 'CFD'     %一致性失败多样性策略
        tempResult=sum(modelResult,2);  %将所有分类器预测结果相加
%         numbers=hist(tempResult, unique(tempResult));
        sumCFD=0;
        for i=1:n   %对每个分类器
            numFialedByNModel=length(find(tempResult==i));
            numFialedByAtLeastOneModel=length(find(tempResult~=n));
            sumCFD=sumCFD+((n-i)/(n-1))*numFialedByNModel/numFialedByAtLeastOneModel;
        end
        diversity=sumCFD;
end
end