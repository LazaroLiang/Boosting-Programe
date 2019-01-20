function [FiltSamples,FiltLables] = FiltEX(samples,lables,weights)

samplesNum=length(weights);
minRate=inf;
flag=-1; %�ж��Ƿ�����˷���Ҫ�������
for iter=1:50
    frequency=zeros(1,samplesNum);
    tempSamples=[];
    tempLables=[];
    for i=1:samplesNum  %��ȡ��������
        randNum=rand(1,1);
        for j=1:samplesNum
            if sum(weights(1:j-1))<=randNum &&sum(weights(1:j))>randNum
                tempSamples=[tempSamples;samples(j,:)];
                tempLables=[tempLables;lables(j,1)];
                frequency(j)=frequency(j)+1;
            end       
        end
    end
    rate=frequency/sum(frequency);
    sumRate=sum((rate'-weights).^2);
    if sumRate<minRate && length(unique(tempLables)) == length(unique(lables))
        minRate=sumRate;
        FiltSamples=tempSamples;
        FiltLables=tempLables;
        flag=1;
    end
end
if flag==-1
    FiltSamples=samples;
    FiltLables=lables;
end
end