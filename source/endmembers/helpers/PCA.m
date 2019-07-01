function [U D mu US] = pca(S)

% Conduct PCA
%
% Input
% - S       : Spectra in column vector form
%
% Output
% - U       : Eigenvectors
% - D       : Eigenvalues
% - US      : S rotated to max. variance.
%
% Copyright (c) 2009: Morten Arngren, ma@imm.dtu.dk, December 2009.

% Extract Principal Components, PCA
[M N]   = size(S);
mu      = mean(S,2);
S_      = bsxfun(@minus, S, mu);
[U D V] = svd(S_*S_'/N);
US      = U'*S;