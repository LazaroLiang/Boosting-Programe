function [diversity]=getDiversity(modelResult,diversityMethod,preDiversity) %计算多样性
%modelResult:所有分类器的预测结果，其中每行代表一个样本，列代表对应分类器分类结果，1表示分类正确，0表示分类错误
%diversityMethod：表示多样性度量方法
%preDiversity：上一轮多样性值
[m,n]=size(modelResult);
switch diversityMethod
    case 'Entropy'  %Entropy多样性策略 
        temp=n-ceil(n/2);
        sumVar=0;
        for i=1:m
            correctClassfierNum=sum(modelResult(i,:));  %计算正确分类的i样本的分类器个数
            miVnar=min(correctClassfierNum,n-correctClassfierNum);
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
    case 'DFM'  %双误度量，成对度量策略，通过计算每一对基分类器多样性，然后计算平均值
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