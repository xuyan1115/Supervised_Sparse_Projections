function [R, R2] = SSP(X, y, bit, sparsity, tol, maxItr, debug)

% ---------- Argument defaults ----------
if ~exist('debug','var') || isempty(debug)
    debug=1;
end
if ~exist('tol','var') || isempty(tol)
    tol=1e-5;
end
if ~exist('maxItr','var') || isempty(maxItr)
    maxItr=1000;
end

alpha = 1e-7
beta = 0.01
lambda = 0.01;
% ---------- End ----------

% initialize with a random rotation
dim = size(X, 2);
R = randn(bit, dim);
B = X * R';
t = (B>0);
B(t) = 1;
B(~t) = -1;

%pre compute pca matrix if bit < dim
if(bit < dim)
    [pc, ~] = eigs(cov(X),bit);
    X_pc = X * pc;   
end

% label matrix N x c
if isvector(y) 
    Y = sparse(1:length(y), double(y), 1); Y = full(Y);
else
    Y = y;
end

i = 0; 
while i < maxItr    
    
    i = i + 1;
    
    %fix R,B,R2 update W
    [W, ~, ~] = RRC(B, Y, lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
   
    % fix B,R,W update R2
    %R20 = R2;
%     y1 = (B + beta*X*R')/(1+beta);
%     %y1 = B;
%     if(bit >= dim)
%         R2 = OrthogonalConstrainOpt(X,y1);
%     else
%         Rtmp = OrthogonalConstrainOpt(X_pc,y1);
%         R2 = Rtmp * pc';
%     end
    %if inv(X) is exist
    R2 = 1/(1+beta) * (beta * R * X' + B') * X * inv(X' * X + 1e-3 * eye(size(X, 2)));
%    R2 = 1/(1+beta) * (beta * R * X' + B') * pinv(X');    
   
    % fix B,W,R2, update R
    R = SparseConstrainOpt(R2, sparsity);
    %R = R2;
    
    % fix R,R2,W update B
    Q = Y * W' + alpha * X * R2';
    B = zeros(size(B));
    for time = 1:10
        Z0 = B;
         for k = 1 : size(B, 2)
            Zk = B; Zk(:,k) = [];
            Wkk = W(k,:); Wk = W; Wk(k,:) = [];
            B(:,k) = sign(Q(:,k) - Zk * Wk * Wkk');
        end
        
        if norm(B - Z0, 'fro') < 1e-6 * norm(Z0, 'fro')
            break
        end
    end
    
%     bias = norm(B-X*R2','fro');   
%     if bias < tol*norm(B,'fro')
%         break;
%    end 
    
%     if norm(R2-R20,'fro') < tol * norm(R20)
%         break;
%     end
    
end

function R = OrthogonalConstrainOpt(X, Y)
    %%% min |XR'-Y|^2, s.t. R'R=I
    data_dim = size(X,2);
    bit_num = size(Y,2);

    [U Sigma V] = svd(X'*Y);
    if(bit_num >= data_dim)
        V = V(:,1:data_dim);
    else
        U = U(:,1:bit_num);    
    end
    R = V*U';
end

function R = SparseConstrainOpt(R2, sparsity)
    n_total = numel(R2);
    n_nonzero = ceil(n_total * (1-sparsity));
    values = abs(R2(:));
    values = sort(values, 'descend');
    thresh = values(n_nonzero);
    R=R2;
    R(abs(R)<thresh) = 0;
end

end
