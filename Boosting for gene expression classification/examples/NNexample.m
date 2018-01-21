%% preprocess
clear;
clc;
close all;
load('handwrite.mat');
thresh = graythresh(X);
X = X > thresh;

%% displaydata
sel = sel(1:100);
displayData(X(sel, :));

%% feature extraction
feature = zeros(size(X,1),14);
for i = 1:size(X,1)
    %特征提取
    [featuretemp,bmg,flag] = getFeature(X(i,:));        
    feature(i,:) = featuretemp;
end
%归一化
[feature,se] = mapminmax(feature);                      
%分离训练数据和测试数据
trainX = [];
testX = [];
trainY = [];
testY = [];
for i = 1:10
    trainX = [trainX;feature((i - 1) * 500 + 1:(i - 1) * 500 + 400,:)];
    testX = [testX;feature((i - 1) * 500 + 401:i * 500,:)];
    trainY = [trainY;y((i - 1) * 500 + 1:(i - 1) * 500 + 400,:)];
    testY = [testY;y((i - 1) * 500 + 401:i * 500,:)];
end

%% neural network
tic;
spread = 0.1;
%建立并训练神经网络
net = newpnn(trainX',ind2vec(trainY'),spread);          
ti = toc;
fprintf('建立网络模型共耗时 %f sec\n', ti);

%% predict
lab0 = net(trainX');
lab = vec2ind(lab0);
rate = sum(lab' == trainY) / length(trainY);
fprintf('训练样本的正确率为\n %d%%\n', round(rate*100));
lab0 = net(testX');
lab = vec2ind(lab0);
rate = sum(lab' == testY) / length(testY);
fprintf('测试样本的正确率为\n %d%%\n', round(rate*100));