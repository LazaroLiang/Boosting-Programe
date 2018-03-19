clc;
clear;
samples=[1,2;3,4;5,6;7,8;9,10;2,1;4,3;6,5;8,7;10,9];
% weight=0.1*ones(10,1);
weight=[0.01;0.1;0.05;0.1;0.1;0.1;0.05;0.05;0.1;0.34]
lables=ones(10,1);
sam=FiltEX(samples,lables,weight)
% allSam=[];
% for i=1:10
%     sam=FiltEX(samples,lables,weight);
%     allSam=[allSam;sam];
% end
% 
% [b m n]=unique(allSam,'rows');
% c=tabulate(n);
% dot=allSam(m(c(:,1)),:);
% num=c(:,2);
% disp(sprintf('%6s%6s%6s','x','y','num'));
% disp([dot num num/100]); 