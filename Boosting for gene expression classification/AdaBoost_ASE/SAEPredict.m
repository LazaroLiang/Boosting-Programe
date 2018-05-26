function [result]=SAEPredict(sae_model,train_set) 
%通过sae模型进行预测
  labels = nnpredict(sae_model, train_set);
%   setSize=length(train_set);
  result=labels;
%   for t=1:setSize
%     if(labels(t,1)==1)
%         result=[result;1];
%     else
%         result=[result;2];
%     end
% end
end