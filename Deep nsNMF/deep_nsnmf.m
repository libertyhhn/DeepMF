function [ Z, H,S, objhistory,constr_err,expl_variance ] = deep_nsnmf ( X, layers, opts )
%  Deep Non-Smooth Nonnegative Matrix Factorization (Deep nsNMF)
%
% The problem of interest is defined as
%
%           min || X - Z_{1}S_{1}...Z_{m}S_{m}H_{m}))||_F^2,
%                              s.t. Z_{i}, S_{i}, H_{m} > 0.
%           where 
%                 m denotes the m-th layer  
% Inputs: 
%       X         : (m x n)  matrix to factorize
%       layers    : [1st 2nd 3th ...] layers sizes
%       opts      : thl,... (hyper-parameters
%
% Outputs:
%        Z     : each layer weighted matrix
%        H_err : final feature matrix
% 
% Reference:
%       J. Yu, G. Zhou, A. Cichocki, and S. Xie,
%       "Learning the Hierarchical Parts of Objects by Deep Non-Smooth Nonnegative Matrix Factorization,"
%       IEEE Access, vol. 6, pp. 58096-58105, Oct. 2018  
%
% Created by Jinshi Yu, 2018
% Modified by Haonan Huang, 2020

num_of_layers = numel(layers);

Z = cell(1, num_of_layers);
H = cell(1, num_of_layers);
S = cell(1, num_of_layers);
    
optsdef=struct('z0',0, 'h0',0, 'TolFun',1e-3, 'verbose',1, 'fast',1, 'thl',[0.1 0.9]);   
if ~exist('opts','var')
    opts=struct;
end
[z0, h0, tolfun, verbose, fastapprox,thl]=scanparam(optsdef,opts);

if  ~iscell(h0)
    for i_layer = 1:length(layers)
        if i_layer == 1
            % For the first layer we go linear from X to Z*H, so we use id
            V = X;
        else 
            V = H{i_layer-1};
        end
        
        if verbose
            display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', i_layer, layers(i_layer), mat2str(size(V))));
        end
                   
            [Z{i_layer}, H{i_layer},S{i_layer}, ~] = ...
                 nsnmf(V, ...
                     layers(i_layer),fastapprox, ...
                     struct('maxiter', 500, ...
                      'thlta0', thl(i_layer)));       
    end

else
    Z=z0;
    H=h0;
    
    if verbose
        display('Skipping initialization, using provided init matrices...');
    end
end

%% Error Propagation
if verbose
    display('Finetuning...');
end
m = size(X,1);
objhistory = cost_function(X, Z, H, S);
timestarted = clock;
iter = 0;

while 1 
    iter = iter+1;
    
     for i = 1:numel(layers)
        if i==1
              B = S{i};
              for t = (i+1):numel(layers)
                  B =B*Z{t}*S{t};
              end
              B = B*H{numel(layers)};
              BBt=B*B';
              numer = X*B';
        
              for t = 1:10
                  Z{i}=Z{i}.*(numer./(Z{i}*(BBt)+ eps(numer)));
              end  
        
        else       
              A = Z{1}*S{1};
             for t = 2:(i-1)
                 A = A*Z{t}*S{t};
             end
             AtA=A'*A;
        
             B = S{i};
             for t = (i+1):numel(layers)
                 B =B*Z{t}*S{t};
             end
             B = B*H{numel(layers)};
             BBt=B*B';
             numer = A'*X*B';
        
             for t = 1:10
                 Z{i}=Z{i}.*(numer./((AtA)*Z{i}*(BBt)+ eps(numer)));
             end
        end
        Z{i} = bsxfun(@rdivide,Z{i},sqrt(sum(Z{i}.^2,1)));
       
        
        if i == numel(layers)
            A = Z{1}*S{1};
            for t = 2:i 
                A = A*Z{t}*S{t};
            end
            numer = A' * X;
            AtA=A'*A;
            for t = 1: 10
            H{i} = H{i} .* (numer ./ (((AtA) * H{i}) + eps(numer)));
            end
        end
     end
        
    if rem(iter,5)==0
	elapsed = etime(clock,timestarted);
    newobj = cost_function(X, Z, H, S);
    objhistory = [objhistory newobj];  

    end
    if iter==800
        break;
    end

    if length(objhistory)>=20
    if objhistory(end-1)>=objhistory(end) && (objhistory(end-1)-objhistory(end) )<= tolfun*max(1,objhistory(end-1))
            display(sprintf('Stopped at %d: objhistory(end): %f, objhistory(end-1): %f', iter, objhistory(end), objhistory(end-1)));
            break;
    end
    end

end
constr_err = objhistory(end)^2;
expl_variance = 1-constr_err/sum(sum(X.^2));

end

function error = cost_function(X, Z, H, S)
    error = norm(X - reconstruction(Z, H, S), 'fro');
end

function [ out ] = reconstruction( Z, H, S )

    out = H{numel(H)};

    for k = numel(H) : -1 : 1
        out =  Z{k} * S{k}*out;
    end

end
