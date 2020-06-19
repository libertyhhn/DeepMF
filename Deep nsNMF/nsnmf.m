function [ Z, H,S, objhistory] = nsnmf ( X, k,fastapprox,opts)
% Nonsmooth nonnegative matrix factorization (nsNMF)
%
% The problem of interest is defined as
%
%           min || X - ZSH||_F^2,
%               s.t. Z, S, H > 0.
%
% Matrix sizes
% X     :                 m x n
% Z     :                 m x num_of_components
% S     : num_of_components x num_of_components
% H     : num_of_components x n
% 
% Do NNSVD initialisation of the init components
% thlta is sparse parameters
% 
% Reference:
%       J. Yu, G. Zhou, A. Cichocki, and S. Xie,
%       "Learning the Hierarchical Parts of Objects by Deep Non-Smooth Nonnegative Matrix Factorization,"
%       IEEE Access, vol. 6, pp. 58096-58105, Oct. 2018  
%
% Created by Jinshi Yu, 2018
% Modified by Haonan Huang, 2020

if fastapprox
    [z0, h0] = NNDSVD(abs(X), k, 0);
else
    z0 = rand(size(X, 1), k);
    h0 = rand(k, size(X,2));
end

optsdef=struct('z0',z0, 'h0',h0, 'bUpdateH',1, 'maxiter',500,'TolFun',1e-3,'thlta0',0.9);
if ~exist('opts','var')
    opts=struct;
end

[z0, h0, bUpdateH, max_iter,tolfun,thlta]=scanparam(optsdef,opts);   
    
S = (1-thlta)*eye(k)+thlta/k*ones(k);

Z = z0;
H = h0;
objhistory = cost_function(X, Z, H, S);

for i = 1:max_iter    
      
       B = S*H;
       numer = X * B';
       for t =1:10
           Z = Z .* (numer ./  ((Z * (B * B')) + eps(numer)));   
       end
       Z = bsxfun(@rdivide,Z,sqrt(sum(Z.^2,1)));
       
       if bUpdateH
            A = Z*S;
            numer = A' * X; 
            for t = 1:10
                H = H .* (numer ./ (((A' * A) * H) + eps(numer)));
            end
        end
       
       if mod(i, 5) == 0
            newobj = cost_function(X, Z, H, S);
            objhistory = [objhistory newobj];
       end
        if length(objhistory)>=20
            if objhistory(end-1)>=objhistory(end) && objhistory(end-1)-objhistory(end)<=tolfun*max(1,objhistory(end-1))
                fprintf('Stopped at %d: objhistory(end)(end): %f, objhistory(end-5): %f\n', i, objhistory(end), objhistory(end-1));
                break;
            end
        end    
end

end
function error = cost_function(X, Z, H, S)
    error = norm(X - reconstruction(Z, H, S), 'fro');
end

function [ out ] = reconstruction( Z, H, S )
        out =  Z * S*H;
end



