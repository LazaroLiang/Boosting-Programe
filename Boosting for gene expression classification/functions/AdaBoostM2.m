function AccuracyRate = AdaBoostM2(trainX,trainY,testX,testY)
tic;
ada = fitensemble(trainX,trainY,'AdaBoostM2',50,'tree');
result = predict(ada,testX);
AccuracyRate = sum(result == testY) / length(testY);
%rate
% fprintf('AdaBoostM1 Accuracy:\n %d%%\n', round(rate*100));
fprintf('AdaBoostM2 Accuracy: %d%%\n', roundn(AccuracyRate,-4));
ti = toc;
fprintf('Time: %f sec\n', ti);