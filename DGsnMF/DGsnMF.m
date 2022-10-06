% Title   : "Deep Graph semi-NMF algorithm and its convergence",submitted to Neurocomputing, 2020. 
% Authors : Haonan Huang
% Affl.   : Guangdong University of Technology, Guangzhou, China
% Email   : libertyhhn@foxmail.com
% 	
% Be noted that we modified from the Deep Semi-NMF implementation provide by the authors as follows:
% "A Deep Semi-NMF Model for Learning Hidden Representations"
% George Trigeorgis, Konstantinos Bousmalis, Stefanos Zafeiriou, Bjoern W. Schuller 
% Proceedings of The 31st International Conference on Machine Learning, pp. 1692¨C1700, 2014 
% Deep Semi-NMF code URL: https://github.com/trigeorgis/Deep-Semi-NMF

function [ Z, H_err] = DGsnMF( X, layers,varargin )
% forward-backward splitting for Deep Graph semi non-negative Matrix Factorization (DGsnMF)
%
% The problem of interest is defined as
%
%           min || X - Z_{1}...Z_{l}H_{l} ||_F^2 + Tr((H_{l})^{T}LH_{l}),
%                                                          s.t. H_{l} > 0
%           where l denotes the l-th of layer , L denotes the graph matrix,Tr denotes the trace 
%           .
% Inputs: 
%       X         : (m x n)  matrix to factorize
%       layers    : [1st 2nd 3th ...] layers sizes
%       varargin  : maxiter,lambdas,...
%
% Outputs:
%        Z     : each layer weighted matrix
%        H_err : final feature matrix


pnames = { ...
    'z0' 'h0' 'maxiter' 'TolFun', ...
    'verbose', 'beta', 'lambdas', 'gnd' ...
};

num_of_layers = numel(layers);

Z = cell(1, num_of_layers);
H = cell(1, num_of_layers);
L = cell(1, num_of_layers);
W = cell(1, num_of_layers);
Di = cell(1, num_of_layers);
LpC = cell(1, num_of_layers);
costtimeIter = [];
dnormMat = [];
nSmp = size(X, 2);
dflts  = {0, 0, 1000, 1e-5, 1,1,1e-3};
[z0, h0, maxiter, tolfun, verbose, beta,lambdas] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});

%% Pre-train 
if  ~iscell(h0)
    for i_layer = 1:length(layers)
        if i_layer == 1
            V = X;
        else 
            V = H{i_layer-1};
        end
        
        if verbose
            fprintf(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...',...
                i_layer, layers(i_layer), mat2str(size(V))));
        end
        
        %construct graph laplacian matrix 
        options = [];
        options.WeightMode = 'Binary'; 
        options.k = 10;
        W{i_layer} = constructW(V',options)';        
        W{i_layer} = lambdas * W{i_layer};
        DCol = full(sum(W{i_layer},2));
        Di{i_layer} = spdiags(DCol,0,nSmp,nSmp);
        L{i_layer} = Di{i_layer} - W{i_layer};       
        if ~issparse(L{i_layer})
            Lp=norm(L{i_layer});   
        else
            Lp=norm(full(L{i_layer}));
        end
        LpC{1,i_layer} = Lp;
        % single layer version of DGsnMF
        [Z{i_layer}, H{i_layer}, ~] = GsnMF(V, layers(i_layer), maxiter, ...
            0.025,L{i_layer},LpC{1,i_layer}); 
    end

else
    Z=z0;
    H=h0;
    
    if verbose
        fprintf('Skipping initialization, using provided init matrices...');
    end
end

dnorm0 = cost_function(X, Z, H,L);
dnorm = dnorm0;

if verbose
    fprintf('#%d error: %f\n', 0, dnorm0);
end

%% Propagation

if verbose
    fprintf('Finetuning...');
end
H_err = cell(1, num_of_layers);
    H_err{numel(layers)} = H{numel(layers)};
    for i_layer = numel(layers)-1:-1:1
        H_err{i_layer} = Z{i_layer+1} * H_err{i_layer+1};
    end   
tStart=tic;
for iter = 1:maxiter  
    for i = 1:numel(layers)
        if i == 1
            HVt=H_err{1}*X'; HHt=H_err{1}*H_err{1}';   % Optimize Z{1} with H fixed
            Z{i} = Ne_NNLS(Z{i},HHt,HVt);
        else
            VHt = X*H_err{i}'; HHt=H_err{i}*H_err{i}'; % Optimize Z{l} with H fixed
            Z{i} = Ne_i_NNLS(Z{i},HHt,VHt,D);
        end
        
        if i == 1
            D = Z{1};
        else
            D = D * Z{i};
        end
        
        if i == numel(layers)
            WtV=D'*X; WtW=D'*D;
            [H{i}]=NNLS_MR(H{i},WtW,WtV,beta,L{i},LpC{i});  % Optimize H with Z fixed
        end
    end
    
    assert(i == numel(layers));
    
    dnorm = cost_function(X, Z, H,L);
   
    if verbose
        if mod(iter, 100) == 0
            fprintf('#%d error: %f\n', iter, dnorm);
        end
    end    
    
    costtime=toc(tStart);   
    costtimeIter = [costtimeIter costtime];

    % stop condition
    if iter>2 && (dnorm0-dnorm <= tolfun*max(1,dnorm0))
        if verbose
            fprintf( ...
                sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', ...
                    iter, dnorm, dnorm0 ...
                ) ...
            );
        end
        break;
    end
    
    dnorm0 = dnorm;
    dnormMat = [dnormMat dnorm];
    H_err{numel(layers)} = H{numel(layers)};
    for i_layer = numel(layers)-1:-1:1
        H_err{i_layer} = Z{i_layer+1} * H_err{i_layer+1};
    end
end
parm.costtimeIter = costtimeIter;
parm.dnormMat = dnormMat;

end

function [H,Grad]=Ne_NNLS(Z,WtW,WtV)
if ~issparse(WtW)
    L=norm(WtW);	% Lipschitz constant
else
    L=norm(full(WtW));
end
% Grad=WtW*Z-WtV;     % orginal Gradient
Grad=Z*WtW-WtV';     
H=Z-Grad/L;    

denom=max(1,sqrt(sum(H.^2)));
ind=find(denom>0);
H(:,ind)=H(:,ind)./(ones(size(H,1),1)*denom(ind));
end

function [H,Grad]=Ne_i_NNLS(H,WtW,VHt,Z)
ZtZ = Z'*Z;
if ~issparse(WtW)
    L=norm(ZtZ)*norm(WtW);	% Lipschitz constant
else
    L=norm(full(ZtZ))*norm(full(WtW));
end
Grad=ZtZ*H*WtW-Z'*VHt;     
H=H-Grad/L;          
denom=max(1,sqrt(sum(H.^2)));
ind=find(denom>0);
H(:,ind)=H(:,ind)./(ones(size(H,1),1)*denom(ind));
end

function [H,Grad]=NNLS_MR(Z,WtW,WtV,beta,Lp,LpC)

if ~issparse(WtW)
    L=norm(WtW)+beta*LpC;	% Lipschitz constant
else
    L=norm(full(WtW))+beta*LpC;
end
Grad=WtW*Z-WtV+beta*Z*Lp;     % Gradient
H=max(Z-Grad/L,0);    % Calculate sequence 'Y'
end

function error = cost_function(X, Z, H,L)
    error = norm(X - reconstruction(Z, H), 'fro')+ trace(H{numel(H)} * L{numel(H)} * H{numel(H)}');
end

function [ out ] = reconstruction( Z, H )

    out = H{numel(H)};

    for k = numel(H) : -1 : 1
        out =  Z{k} * out;
    end

end
