function AccuracyRate = SVMDecision(trainX,trainY,testX,testY)
tic;            %¿ªÆôÊ±ÖÓ
model = svmtrain(trainX,trainY);
[result] = svmclassify(model,testX);
AccuracyRate = sum(testY== result)/length(testY);
% [result] = svmpredict( testX,testY, model);
fprintf('SVM Accuracy:%d\n',AccuracyRate);
ti = toc;
fprintf('Time: %f sec\n', ti);