function L = getL(training_feats,H_train,k,lamda)
%% Constructing a supervised graph and calculate Laplacian matrix
% Input:
%       training_feats  -each column is a data point
%       H_train         -a one-hot binary matrix (size: nLabel * nSmp)
%       k               -number of nearest neighbors
%       lamda           -regularization parameter, i.e., Ww-lamda*Wb
% Output:
%       L               -Laplacian matrix


%% initialization
[nLabel, nSmp] = size(H_train); % [# of classes, # of samples]
Ww = zeros(nSmp,nSmp);
Wb = ones(nSmp,nSmp);
for idx=1:1:nLabel
    classIdx = find(H_train(idx,:)==1);
    Ww(classIdx,classIdx) = 1; 
    Wb(classIdx,classIdx) = 0; 
end


%% distance and weighting
Dist = EuDist2(training_feats',[],1); % Euclidean distance
if k > 0
    WwIdx = sparse(Ww==0);
    Ww(WwIdx) = 1e10;
    [~, idx] = sort(Ww.*Dist,2); % sort each row ascend
    Ww(WwIdx) = 0;
    clear WwIdx  % clear useless variable
    idx = idx(:,2:k+1); % default: not self-connected
    G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
    % G is a square matrix, indicates same-class nearest neighbors
    % i^th row of G: Among the other (nSmp-1) samples, which belongs to 
    % the i^th sample's k nearest neighbors  =1: belong; =0: not belong
    G = max(G,G');
    Ww = Ww.*G;
    clear G idx  % clear useless variable
    
    WbIdx = sparse(Wb==0);
    Wb(WbIdx) = 1e10;
    [~, idx] = sort(Wb.*Dist,2); % sort each row ascend
    Wb(WbIdx) = 0;
    clear WbIdx  % clear useless variable
    idx = idx(:,1:k);    
    G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
    % G is a square matrix, indicates different-class nearest neighbors
    % i^th row of G: Among the other (nSmp-1) samples, which belongs to 
    % the i^th sample's k nearest neighbors  =1: belong; =0: not belong
    G = max(G,G');
    Wb = Wb.*G;
    clear G idx  % clear useless variable
end
Ww = Ww.*exp(-Dist); % within-class
Wb = Wb.*exp(-Dist); % between-class
clear Dist  % clear useless variable
Ww = max(Ww,Ww'); % guarantee symmetry
Wb = max(Wb,Wb'); % guarantee symmetry


%% obtain the Laplacian matrix
L = diag(sum(Ww-lamda*Wb,2))-(Ww-lamda*Wb); % Laplacian Matrix
clear Ww Wb  % clear useless variable
L = full(max(L,L')); % guarantee symmetry