function AccuracyRate = KNN(trainX,trainY,testX,testY)
tic;
knn=fitcknn(trainX,trainY);
result = predict(knn,testX);
AccuracyRate = sum(result == testY) / length(testY);
%rate
% fprintf('AdaBoostM1 Accuracy:\n %d%%\n', round(rate*100));
fprintf('KNN Accuracy:\n %d%%\n', roundn(AccuracyRate,-4));
ti = toc;
fprintf('Time: %f sec\n', ti);