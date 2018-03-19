%testPCA
clear;clc;
load .\data\original_data\colon.mat
data=Sample';
id=[45,51,55,56];
data(id,:)=[];
% noLableData=data(:,1:end-1);
%  [coeff, score, latent, tsquared, explained] = pca(noLableData);
%  feature_after_PCA=score(:,1:20);
%  data=[feature_after_PCA,data(:,end)];