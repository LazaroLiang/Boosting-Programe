function [er,bad]=CNNTest(modle,test_set,test_label)
classNums=length(unique(test_label));
testSetSize=length(test_label);
test_label= test_label(:,1);
test_label = [test_label(:,:) zeros(testSetSize,classNums-1)];
for t=1:testSetSize
    l=test_label(t,1);
    if(l~=1)
        test_label(t,1)=0;
        test_label(t,l)=1;
    end
%     if()
%     if(test_label(t,1)==2)
%         test_label(t,1)=0;
%         test_label(t,2)=1;
%     end
end
[er, bad] = cnntest(modle, test_set, test_label);
end

