function [FiltSamples,FiltLables] = FiltEX(samples,lables,weights)
FiltSamples=[];
FiltLables=[];
samplesNum=length(weights);
for i=1:samplesNum  %?????????
    randNum=rand(1,1);
    for j=1:samplesNum
        if sum(weights(1:j-1))<=randNum &&sum(weights(1:j))>randNum
            FiltSamples=[FiltSamples;samples(j,:)];
            FiltLables=[FiltLables;lables(j,1)];
        end       
    end
end
end