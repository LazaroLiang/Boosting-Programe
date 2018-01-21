function result = NeuroNetwork(trainX,trainY,testX,testY)
tic;
spread = 0.1;
%建立并训练神经网络
net = newpnn(trainX',ind2vec(trainY'),spread);          
ti = toc;
fprintf('建立网络模型共耗时 %f sec\n', ti);

lab0 = net(testX');
result = vec2ind(lab0);
rate = sum(result' == testY) / length(testY);
fprintf('Neruo Network:\n %d%%\n', round(rate*100));