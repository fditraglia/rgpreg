
% nnn = 100;
% x1 = randn(nnn,1);
% % x2 = randn(nnn,1);
% tau = 1;
% theta = 1/2;
%
% tic
% C = gp_K(x1,x1,tau,theta);
% toc



% Single index model
clc; clear all;

%% DGP - Choi et al.

% parameters
% sig2 : noise
% bet  : coeff
% tau  : hyper on GP
% theta: hyper on GP
sig2  = 0.3;
bet   = [0.45; sqrt(1-0.45^2)];
tau   = sig2*2;
theta = 0.1;
nobs  = 50;

% simulated data
X    = -3+(5+3)*rand(nobs,2);
T    = X*bet;
etaT = 0.1*T + sin(0.5*T).^3;
Y    = etaT + sqrt(sig2)*randn(nobs,1);

% plot(T,Y, '*')

Cn     = gp_K(X*bet,X*bet,tau,theta);
all(eig(Cn+sig2*eye(nobs))>0)
all(eig(Cn)>0)

% test cov function is psd
theta_set = linspace(0.001,0.5,100);
test = zeros(100,1);
for i=1:1:length(theta_set)
Cn     = gp_K(X*bet,X*bet,tau,theta_set(i));
test(i,1) = all(eig(Cn)>0);
end

%% Estimation - MAP
% Maximize the marginal likelihood to get bet, tau, theta, sigma

% initial guess
sig20  = sig2;
bet0   = bet(1);
theta0 = theta;
tau0   = tau;

% objective function
para0 = [sig20;bet0;theta0;tau0];
% para0 = [0.05, 0.5234, 0.1846, 0.4846];
option.Display = 'iter';
objfun = @(para) objfun_sim(Y,X,para);
[x,fval] = fminsearch(objfun, para0,option);

%% Estimation - Gibbs sampler

% simulation setting
nsimul = 1000;

% prior
pr.sig2.T = 2;
pr.sig2.V = sig2*(pr.sig2.T-1); %center around true sig2

% initialization of gibbs sampler (from marginal likelihood opt)
% sig2old  = sig2;
% betold   = bet;
% thetaold = theta;
% tauold   = tau;

sig2old  = x(1);
betold   = [x(2); sqrt(1-x(2)^2)];
thetaold = x(end-1);
tauold   = x(end);

% MH initialization
% bet
betold(1) = betold(1)*0.1; %perturb a bit
betold(2) = sqrt(1-betold(1)^2);

% matrix to store
mh_eta   = zeros(nsimul,nobs);
mh_sig2  = zeros(nsimul,1);
mh_bet   = zeros(nsimul,2);
mh_theta = zeros(nsimul,1);
mh_tau   = zeros(nsimul,1);

mh_betrej = zeros(nsimul,1);
mh_betr = zeros(nsimul,1);

% Gibbs sampler starts here
for simulind = 1:1:nsimul
    %     sig2old
    % -----------------------------------------------------------------------
    % eta update - multivariate normal
    Cn       = gp_K(X*betold,X*betold,tauold,thetaold);
    temp_C   = Cn/(Cn+sig2old*eye(nobs));
    CnInvKer = (temp_C + temp_C')/2; %symmetrize... numerical issue here
    eta_m    = CnInvKer*Y;
    eta_V    = sig2old*CnInvKer;
    etaold   = mvnrnd(eta_m',eta_V)';
    
    % Vinv = (sig2old*eye(nobs))^(-1) + Cn^(-1);
    % V = inv(Vinv);
    % m = Vinv\(Y/sig2old);
    % mvnrnd(m',V)
    
    % -----------------------------------------------------------------------
    % sig2 update - inverse gamma
    postT   = pr.sig2.T+1+nobs/2;
    postV   = pr.sig2.V + 1/2*sum((Y-etaold).^2);
    sig2old = gamrnd(postT, postV^(-1))^(-1);
    
    % -----------------------------------------------------------------------
    % bet update: RW-MH with flat prior
    % tuing para for MH
    Cbet = 0.001;
    
    % step 1: propose bet
    temp_bet = betold(1) + Cbet*randn(1,1);
    betnew = [temp_bet; sqrt(1-temp_bet^2)];
    
    % step 2: calculate MH ratio
    Cn_new = gp_K(X*betnew,X*betnew,tauold,thetaold);
    post_new = mvnpdf(etaold', zeros(1,nobs), Cn_new);
    
    Cn_old = gp_K(X*betold,X*betold,tauold,thetaold);
    post_old = mvnpdf(etaold', zeros(1,nobs), Cn_old);
    
    % step 3: reject/accept
    betr = (log(post_new)-log(post_old));
    if rand < exp(betr) % accept
%         pause
        betold = betnew;
        betrej = 0;
    else
        betrej = 1;
    end
    
    
    % -----------------------------------------------------------------------
    % Store matrix
    mh_eta(simulind,:)    = etaold';
    mh_sig2(simulind,1)   = sig2old;
    mh_bet(simulind,:)    = betold';
    mh_betrej(simulind,1) = betrej;
    mh_betr(simulind,1)   = betr;
end



%% Posterior mean
plot(T,mean(mh_eta,1),'*')
hold on
plot(T,quantile(mh_eta,0.05,1),'r*');
plot(T,quantile(mh_eta,0.95,1),'r*');
plot(T,Y, 'k*')
plot(T,etaT,'g*')
hold off
legend('posterior mean', 'lower qt', 'high qt', 'data', 'true');










