clc;clear;

learners=30;
iters=20;
rows=2;
cols=4;

load .\data\original_data\Brain_Tumor1.mat
data(:,1)=data(:,1)+1;
data=rot90(data);
subplot(rows,cols,6)
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(data,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate);
title('Brain');
xlabel('学习器个数');
ylabel('集成分类错误率');

load .\data\original_data\Leukemia2.mat
data(:,1)=data(:,1)+1;
data=rot90(data);
subplot(rows,cols,7)
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(data,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate);
title('Leukemia');
xlabel('学习器个数');
ylabel('集成分类错误率');

load .\data\original_data\colon.mat
subplot(rows,cols,1)
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(Sample,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate);
title('colon');
xlabel('学习器个数');
ylabel('集成分类错误率');

load .\data\original_data\prostate.mat
subplot(rows,cols,2)
% learners=3;
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(Sample,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate)
title('prostate');
xlabel('学习器个数');
ylabel('集成分类错误率');

load .\data\original_data\mit.mat
subplot(rows,cols,3)
% learners=3;
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(Sample,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate)
title('mit');
xlabel('学习器个数');
ylabel('集成分类错误率');

load .\data\original_data\nci64.mat
subplot(rows,cols,4)
% learners=3;
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(Sample,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate)
title('nci64');
xlabel('学习器个数');
ylabel('集成分类错误率');

load .\data\original_data\lymphoma.mat
subplot(rows,cols,5)
% learners=3;
for weak_learner_n=1:1:learners
    error_rate(weak_learner_n)=PCA_SAE_AdaBoost(Sample,60,weak_learner_n,5,iters);
end
x=1:1:learners;
plot(x,error_rate)
title('lymphoma');
xlabel('学习器个数');
ylabel('集成分类错误率');

