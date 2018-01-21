function result = BayesNaive(trainX,trainY,testX,testY)
tic;            %¿ªÆôÊ±ÖÓ
model = NaiveBayes.fit(trainX,trainY);
result = model.predict(testX);
rate = sum(result == testY) / length(testY);
fprintf('Bayes accuracy: \n %d%%\n', round(rate*100));
ti = toc;
fprintf('Time: %f sec\n', ti);