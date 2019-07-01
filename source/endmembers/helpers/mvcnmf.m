function [W, time] = MVCNMF(X, K)
t0 = clock;
% Non-Negative Matrix Factorisation with Minimum Volume Constraint
%
% Linear model:     X = W*H + e,    s.t. X>=0, W>=0, H>=0
%
% Input:
% - X (M,N) : M (dimensionality) x N (samples) non negative input matrix
% - K       : Number of components
% - opt
%   .alpha  : Weight for Min. Volume Constraint
%   .lambda : Weight for sparsity on H
%   .reg    : Type of reg. on W (default = 'WtW')
%             - 'WtW', det(W'W), Schachtner.
%             - 'MVC', minimum volume constraint, Miao et al.
%             - 'Dist', Eucldian distance to mean center.
%             - 'Sparse', L2-norm regularization.
%   .W.init : Initialization of W (optional)
%   .W.step : Initial stepsize for W (optional)
%   .W.lock : No update of W (optional)
%   .W.nn   : W non-negative or not (optional)
%   .H.init : Initialization of H (optional)
%   .H.step : Initial stepsize for H (optional)
%   .U      : Projection vectors for plotting only, e.g. principal
%             components.
%   .tol    : Convergence criteria tolerance
%   .maxIt  : Maximum number of iterations to run
%   .doPlot : Plot or not (slow)
%
% Output:
% - W       : Vertices/Endmembers,   M x K matrix
% - H       : Fractional abundances, K x N matrix
%
% Copyright (c) 2009: Morten Arngren, ma@imm.dtu.dk, September 2009.
%
% V1.0 March 2010   : Initial issue.
%


%%
try opt.alpha;      catch opt.alpha      = 0;     end  % Volume reg. on W.
try opt.lambda;     catch opt.lambda     = 0;     end  % Sparsity for H.
try opt.W.init;     catch opt.W.init     = [];    end  % Init. of W.
try opt.W.true;     catch opt.W.true     = 0;     end  % For plotting only.
try opt.W.step;     catch opt.W.step     = 1e-3;  end
try opt.W.lock;     catch opt.W.lock     = [];    end  % Update W or not, 1 = not
try opt.W.nn;       catch opt.W.nn       = 1;     end  % W nonnegative or not.
try opt.H.step;     catch opt.H.step     = 1e-3;  end
try opt.H.init;     catch opt.H.init     = 0;     end  % Init. of H.
try opt.maxIt;      catch opt.maxIt      = 1000;  end
try opt.maxItLocal; catch opt.maxItLocal = 100;   end
try opt.reg;        catch opt.reg        = 'WtW'; end  % Type of regularization
try opt.U;          catch opt.U          = 0;     end  % PC vectors for plotting

%% Initialise
regWAll  = 0;
opt.Wmis = 0;
opt.Hmis = 0;
[D,N]    = size(X);
Xscale   = sum(sum(X));


% Init W or H from outside
if ~isempty(opt.W.init)
  W(:,1:size(opt.W.init,2)) = opt.W.init;
else
  W = X(:,ceil(N*rand(K,1)));  
end
if opt.H.init
  H = opt.H.init;
else
  H = rand(K,N);  
end

Rscale  = sum(sum(W*H));
sqrnorm = sqrt(Rscale/Xscale);
H       = H/sqrnorm;
W       = W/sqrnorm;
H       = H ./ repmat(sum(H),K,1);       % Normalize columns to unit length

XX2     = sum(sum(X.^2));
meanX   = mean(X,2);

% Calc. eigenvectors U
if ~opt.U || strcmp(lower(opt.reg), 'miao') || strcmp(lower(opt.reg), 'mvc')
  [U D mu UX] = pca(X);
  U           = U(:,1:K-1);
end
meanX = repmat(mean(X,2), 1, K);

tau = opt.alpha; J=0;
% Constraint
switch lower(opt.reg)
  case 'sparse'
    J   = norm(W,2);
  case {'mvc', 'miao'}
    tau = tau/factorial(K-1);
    B   = [zeros(1,K-1); eye(K-1)];
    C   = [ones(1,K); zeros(K-1,K)];
    BU  = B*U';
    Z   = C + BU*(W - meanX);
    J   = (det(Z))^2;
  case {'vol_pp', 'wtw'}
    J   = det(W'*W);
  case {'dist'}
    W_hat = W - mean(W,2)*ones(1,K);
    J     = sum(sum(W_hat.^2));       % Squared euclidian length of vectors
end

% Only for plotting during iterations.
% if opt.U, U = opt.U; end;
% UX      = U'*X;
% UW_true = U'*opt.W.true;

% Calculate initial error
errOld = XX2 - 2*sum(sum(W'*X.*H)) + sum(sum( W'*W.*(H*H') ));
errOld = 0.5*(errOld + tau*J);
Err    = errOld;

%% Iterate
n = 1; deltaErr = inf; Outerloop = 1;
while Outerloop 
  
  %%%%%% Update H
  WtX = W'*X;
  WtW = W'*W;
  
  grad = WtW*H - WtX + opt.lambda;
  grad = grad - repmat(sum(H.*grad),K,1);
  
  loop = 1; itLocal = 0;
  while loop
    H_       = H - opt.H.step*grad;
    H_(H_<0) = 0;           % Project negative elements to positive orthant
    H_       = H_ ./ repmat(sum(H_),K,1); % normalize columns to unit length
    err      = XX2 - 2*sum(sum(WtX.*H_)) + sum(sum( WtW.*(H_*H_') ));
    err      = 0.5*(err + tau*J);
    
    if err < errOld
      opt.H.step = 1.2*opt.H.step;
      H = H_; loop = 0;
    else
      opt.H.step = 0.5*opt.H.step;
    end
    itLocal = itLocal + 1;
    if itLocal > opt.maxItLocal, loop = 0; end
  end
  
  % Save current error
  errOld = err;
  
  %%%%%% Update W
  if isempty(opt.W.lock)
    loop = 1; itLocal = 0;
    HHt  = H*H';
    XHt  = X*H';
    grad = W*HHt - XHt;
    
    switch lower(opt.reg)
      case 'sparse'
        J    = norm(W,2);
        grad = grad + tau*W;
      case {'mvc', 'miao'}
        Z    = C + BU*(W - meanX);
        J    = (det(Z))^2;
        grad = grad + tau*J*BU'*inv(Z)';   % NMF-MVC by Miao et al.
      case {'vol_pp', 'wtw'}
        WtW  = W'*W;
        J    = det(WtW);
        grad = grad + (tau*2*J*W)/WtW;
      case {'dist'}
        W_hat = W - mean(W,2)*ones(1,K);
        J     = sum(sum(W_hat.^2));   % Squared euclidian length of vectors
        grad  = grad + tau*2*W_hat;
    end
    
    while loop
      W_ = W - opt.W.step*grad;
      if opt.W.nn
        W_(W_<0) = 0;       % Project negative elements to positive orthant
      end
      WtW = W_'*W_;
      
      % Calc. new regulation term for error estimation.
      switch lower(opt.reg)
        case 'sparse'
          J   = norm(W_,2);
        case {'mvc', 'miao'}
          Z   = C + BU*(W_ - meanX);
          J   = (det(Z))^2;
        case {'vol_pp', 'wtw'}
          J   = det(WtW);
        case {'dist'}
          W_hat = W_ - mean(W_,2)*ones(1,K);
          J     = sum(sum(W_hat.^2)); % Squared euclidian length of vectors
      end

      % Calc. error of new step
      err = XX2 - 2*sum(sum(W_.*XHt)) + sum(sum( WtW.*HHt ));
      err = 0.5*(err + tau*J);
      
      if err < errOld
        opt.W.step = 1.2*opt.W.step;
        W = W_; loop = 0;
      else
        opt.W.step = 0.5*opt.W.step;
      end
      itLocal = itLocal + 1;
      if itLocal > opt.maxItLocal, loop = 0; end
    end
  end
  
  % Cals. errors
  deltaErr = Err-err;
  Err      = err;
  errOld   = err;
  n        = n + 1;
  regWAll  = [regWAll J];
  
  % Is error low enough to stop?
  Outerloop =  n<opt.maxIt;
  
  
end
opt.it = n;
time = etime(clock,t0);
end
