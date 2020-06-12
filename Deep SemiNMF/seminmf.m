function [Z, H, dnorm] = seminmf( X, k, varargin )
% Matrix sizes
% X: m x n
% Z: m x num_of_components
% H: num_of_components x num_of_components
% 
% Reference:
%       C.H.Q. Ding, T. Li, M. I. Jordan,
%       "Convex and semi-nonnegative matrix factorizations,"
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.1, 2010.
% 
% Modified by Haonan Huang, 2020

pnames = {'z0' 'h0' 'bUpdateH' 'maxiter' 'TolFun' 'bUpdateZ' 'verbose' 'fast'};

h0 = rand(k, size(X, 2)); 

dflts  = {0, h0, 1, 500,  1e-5, 1, 1,1};

[Z, H, bUpdateH, max_iter, tolfun, bUpdateZ, verbose,  fastapprox] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});

if fastapprox
    H = LPinitSemiNMF(X, k); 
end

if length(Z) == 1
    Z = X * pinv(H);
end

dnorm = norm(X - Z * H, 'fro');

for i = 1:max_iter
    
    if bUpdateZ
        try
            Z = X * pinv(H);
        catch
            display('Error inverting');
        end
    end
    
    A = Z' * X;
    Ap = (abs(A)+A)./2;
    An = (abs(A)-A)./2;
    
    B = Z' * Z;
    Bp = (abs(B)+B)./2;
    Bn = (abs(B)-B)./2;
    
    if bUpdateH
        H = H .* sqrt((Ap + Bn * H) ./ max(An + Bp * H, eps));
    end
      
    if mod(i, 10) == 0 || mod(i+1, 10) == 0 
        
        s = X - Z * H;
        dnorm = sqrt(sum(s(:).^2));
        % dnorm = norm(gX - Z * H, 'fro');
        
        if mod(i+1, 10) == 0
            dnorm0 = dnorm;
            continue
        end

        if mod(i, 100) == 0 && verbose
            display(sprintf('...Semi-NMF iteration #%d out of %d, error: %f\n', i, max_iter, dnorm));
        end

        if 0 && exist('dnorm0')
            assert(dnorm <= dnorm0, sprintf('Rec. error increasing! From %f to %f. (%d)', dnorm0, dnorm, k));
        end

        % Check for convergence
        if exist('dnorm0') && dnorm0-dnorm <= tolfun*max(1,dnorm0)
            if verbose
                display(sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', i, dnorm, dnorm0));
            end
            break;
        end
     
    end
end

