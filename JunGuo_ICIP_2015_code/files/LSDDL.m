function D = LSDDL(Y,Dinit,Xinit,L,alpha,gamma)
% 'LSDDL.m' implements LSDDL in Algorithm 1
% Input:
%       Y       -each column is a training sample
%       Dinit   -initialized dictionary
%       Xinit   -initialized codes
%       L       -Laplacian matrix
%       alpha   -regularization parameter for alpha*(Ww-lamda*Wb)
%       tau     -regularization parameter for ||x||_2^2
% Output:
%       D       -learned dictionary


if size(Y,2)~=size(Xinit,2)
    error('The sizes of Xinit and training_feats are mismatching!');
end

Dinit = normcols(Dinit); % guarantee unit l2-norm for each atom
X = zeros(size(Xinit));
err = [eps];
circleID = 1;
while circleID <= 1e3   % maximum number of iterations
    Temp = Xinit * L;    
    for idx = 1:1:size(Y,2) % update X
        X(:,idx) = (Dinit'*Dinit+(alpha*L(idx,idx)+gamma)*eye(size(Dinit,2)))\(Dinit'*Y(:,idx)-alpha*(Temp(:,idx)-L(idx,idx)*Xinit(:,idx)));
    end
    D = Dinit;
    Dinit = Y*X'/(X*X'+5e-8*eye(size(X,1))); % update D
    err = [err norm(Y-Dinit*X,'fro')];
    if ( err(end)>=err(end-1) ) && ( length(err)>=3 )
        Dinit = D;
        break;
    end
    if ( err(end) <= 1e-6 ) || ( abs(err(end)-err(end-1)) <= 1e-6 ) || ( max(max(abs(Dinit-D))) <= 1e-6 ) % stop
        break;
    end
    Xinit = X;
    circleID = circleID + 1;
end
D = normcols(Dinit);