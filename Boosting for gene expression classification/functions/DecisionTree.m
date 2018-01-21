function result = DecisionTree(trainX,trainY,testX,testY)
tic;
x = trainX;
y = num2str(trainY);
y = mat2cell(y,ones(size(trainY)));
% ������������
t = classregtree(x,y);
% ����������
result = eval(t,testX);
% ��÷�����
result = cell2mat(result);
result = str2num(result);
rate = sum(result == testY) / length(testY);
fprintf('Decision Tree:\n %d%%\n', round(rate*100));
% ��ʾ������
treedisp(t);
ti = toc;
fprintf('Time: %f sec\n', ti);