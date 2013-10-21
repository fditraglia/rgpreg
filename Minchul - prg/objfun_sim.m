function loglik = objfun_sim(Y,X,para)
% Marginal likelihood for SIM 
% Input: Y,X,para
% Output: Log-lik
% date: 2013/10/18

% recover parameter
sig2  = para(1);
bet   = [para(2); sqrt(1-para(2)^2)];
theta = para(end-1);
tau   = para(end);

nobs  = size(Y,1);

% loglikelihood
% Cn     = gp_K(X*bet,X*bet,tau,theta);
% Cin    = sig2*eye(nobs) + Cn;
% ploglik = -1/2*log(det(Cin)) + -1/2*(Y'*(Cin\Y)); %log marginal likelihood
% 
% loglik = -ploglik; %negative loglik

% negative log likelihood (RW chapter 2)
Cn     = gp_K(X*bet,X*bet,tau,theta);
Cin    = sig2*eye(nobs) + Cn;
L      = chol(Cin, 'lower');
alp    = L'\(L\Y);
loglik = -( -1/2*Y'*alp -sum(log(diag(L)))  );

% contraint ...
if sig2<0 %positive error variance
    loglik = inf;
end

% para
% chol(Cn)
% if ~all(eig(Cn+sig2*eye(nobs))>0) %postive definite Cn
%     disp('binds');
%     loglik = inf;
% end

% if ~all(eig(Cn)>0) %postive definite Cn
%     disp('binds');
%     loglik = inf;
% end

% loglik
% para