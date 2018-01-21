function result = AdaBoost(trainX,trainY,testX,testY)
tic;
ada = fitensemble(trainX,trainY,'AdaBoostM1',2000,'Discriminant');
result = predict(ada,testX);
rate = sum(result == testY) / length(testY);
fprintf('AdaBoostM1 Accuracy:\n %d%%\n', round(rate*100));
ti = toc;
fprintf('Time: %f sec\n', ti);