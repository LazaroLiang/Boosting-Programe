clear;clc;
load .\data\original_data\Brain_Tumor1.mat
data(:,1)=data(:,1)+1;
data=rot90(data);

PCA_SAE_AdaBoost(data,65,20,5,20)

% learners=50;
% iters=20;
% rows=2;
% cols=4;
% load .\data\original_data\Brain_Tumor1.mat
% data(:,1)=data(:,1)+1;
% data=rot90(data);
% % subplot(rows,cols,6)
% for weak_learner_n=1:1:learners
%     error_rate(weak_learner_n)=PCA_SAE_AdaBoost(data,60,weak_learner_n,5,iters);
% end
% x=1:1:learners;
% plot(x,error_rate);
% title('Brain');
% xlabel('学习器个数');
% ylabel('集成分类错误率');
