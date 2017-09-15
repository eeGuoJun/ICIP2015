function [Dinit,Xinit] = INITdic(trData,H_train,dictsize)
% initialization for D and X
% Input:
%       trData   -each column is a training sample
%       H_train  -a one-hot binary matrix (size: nLabel * nSmp)
%       dictsize -number of columns in the dictionary
% Output:
%       Dinit    -initialized dictionary
%       Xinit    -initialized codes


iterations = 34;
sparsitythres = 30;
addpath(genpath('.\ksvdbox'));
addpath(genpath('.\OMPbox'));
numClass = size(H_train,1); 
numPerClass = round(dictsize/numClass); 
Dinit = [];
dictLabel = [];
for classid=1:numClass
    col_ids = find(H_train(classid,:)==1);
    data_ids = find(colnorms_squared_new(trData(:,col_ids)) > 1e-6);
    perm = [1:length(data_ids)]; 
    % KSVD for each class-specific sub-dictionaries
    Dpart = trData(:,col_ids(data_ids(perm(1:numPerClass))));
    para.data = trData(:,col_ids(data_ids));
    para.Tdata = sparsitythres;
    para.iternum = iterations;
    para.memusage = 'high';    
    para.initdict = normcols(Dpart);   
    [Dpart,~,~] = ksvd(para,'');
    Dinit = [Dinit Dpart];    
    labelvector = zeros(numClass,1);
    labelvector(classid) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
end
params.data = trData;
params.Tdata = sparsitythres;
params.iternum = iterations;
params.memusage = 'high';
params.initdict = normcols(Dinit);
[Dinit,Xinit,~] = ksvd(params,'');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function YY = colnorms_squared_new(XX)  % only for save memory
YY = zeros(1,size(XX,2));
blocksize = 2000;
for i = 1:blocksize:size(XX,2)
    blockids = i : min(i+blocksize-1,size(XX,2));
    YY(blockids) = sum(XX(:,blockids).^2);
end