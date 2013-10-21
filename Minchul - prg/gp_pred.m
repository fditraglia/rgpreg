function [Zm, ZV] = gp_pred(X, Y, Xstar, theta, tau, sig2)
% GP prediction with SE kernel
% X,Y   : data
% Xstar : points to be predicted
% theta, tau: hyperparameters
% sig2  : noise variance

nobs = size(X,1);

Kxx = gp_K(X, X, tau, theta);
Kxz = gp_K(X, Xstar, tau, theta);
Kzz = gp_K(Xstar, Xstar, tau, theta);
Kzx = Kxz';

A   = Kxx + sig2*eye(nobs);
L   = chol(A)';
alp = (L'\(L\Y));
nu  = L\Kxz;
Zm  = Kzx*alp;
ZV  = Kzz - nu'*nu;


