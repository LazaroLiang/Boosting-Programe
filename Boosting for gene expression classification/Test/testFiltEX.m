clc;
clear;
samples=[1,2;3,4;5,6;7,8;9,10;2,1;4,3;6,5;8,7;10,9];
% weight=0.1*ones(10,1);
weight=[0.01;0.1;0.05;0.1;0.1;0.1;0.05;0.05;0.1;0.34]
sam=FiltEX(samples,weight)
