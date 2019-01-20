clear;clc;
load .\data\original_data\Leukemia1.mat
data(:,1)=data(:,1)+1;
data=rot90(data);

PCA_SAE_AdaBoost(data,65,20,5,2)