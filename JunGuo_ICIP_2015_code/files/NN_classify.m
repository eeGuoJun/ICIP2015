function accuracy = NN_classify(D,TrainData,TtData,H_train,H_test,tau)
% 'NN_classify.m' implements classification by 1-NN classifier
% Input:
%       D         -learned dictionary
%       TrainData -each column is a training sample
%       TtData    -each column is a testing sample
%       H_train   -one-hot binary matrix (size: nClass * nTrainingSample)
%       H_test    -one-hot binary matrix (size: nClass * nTestingSample)
%       tau       -regularization parameter for ||x||_2^2
% Output:
%       accuracy  -classification accuracy


%% coding
TestData = (D'*D+tau*eye(size(D'*D)))\(D'*TtData);
clear TtData      % clear useless variable
TrnData = (D'*D+tau*eye(size(D'*D)))\(D'*TrainData);
clear TrainData D % clear useless variable

%% classify
TrnData = TrnData'; 
[~,TrnLabel] = max(H_train);
TrnLabel = TrnLabel';
clear H_train     % clear useless variable
TestData = TestData';
[~,TestLabel] = max(H_test);
TestLabel = TestLabel';
clear H_test      % clear useless variable
% here we use 1-NN classifer (users can try other classifiers)
prediction = knnclassify(TestData,TrnData,TrnLabel,1,'euclidean','nearest');
matchVec = find(prediction==TestLabel);
accuracy = length(matchVec)/length(TestLabel); % compute Accuracy