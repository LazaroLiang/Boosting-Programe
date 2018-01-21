function AccuracyRate = RandomForest(trainX,trainY,testX,testY)
tic;
nTree = 500;
y = num2str(trainY);
y = mat2cell(y,ones(size(trainY)));
forest = TreeBagger(nTree,trainX,y);
temp = predict(forest,testX);
result = zeros(size(temp));
for i = 1:length(temp) 
    result(i) = str2num(cell2mat(temp(i)));
end
AccuracyRate = sum(result == testY) / length(testY);
fprintf('RandomForest:\n %d%%\n', round(AccuracyRate*100));
ti = toc;
fprintf('Time: %f sec\n', ti);