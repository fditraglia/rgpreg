
% Regression
clc; clear all; close all;

%% DGP - Nonlinear regression

% parameters
% sig2 : noise
% bet  : coeff
% tau  : hyper on GP
% theta: hyper on GP
sig2  = 1;
nobs  = 70;

theta = 7.83; %hyperparameter in kernel
tau   = 3; %hyperparameter in kernel

% simulated data
X  = -5+10*rand(nobs,1);
Y  = 2*sin(X) + sqrt(sig2)*randn(nobs,1);
Yt = 2*sin(X); %without noise

plot(X,Y, '*')

Cn     = gp_K(X,X,tau,theta);
all(eig(Cn+sig2*eye(nobs))>0)
all(eig(Cn)>0)

% test cov function is psd
theta_set = linspace(0.1,5,100);
test = zeros(100,1);
for i=1:1:length(theta_set)
Cn     = gp_K(X,X,tau,theta_set(i));
test(i,1) = all(eig(Cn)>0);
end
plot(theta_set, test); ylim([-0.1, 1.1])

%% Housekeeping

% simulation setting
nsimul = 10000;

% -----------------------------------------------------------------------
% MH tuning parameters and prior
% -----------------------------------------------------------------------
    % sig2
    pr.sig2.T = 2;
    pr.sig2.V = sig2*(pr.sig2.T-1); %center around true sig2
    pr.sig2.fun = @(t) gampdf(1/t, pr.sig2.T, pr.sig2.V/2);
    
    % tau
    Ctau       = 140/nobs;
    pr.tau.m   = 1;
    pr.tau.V   = 4;
    pr.tau.fun = @(t) normpdf(t, pr.tau.m, sqrt(pr.tau.V));

    % theta
    Ctheta       = 10/nobs;
    pr.theta.m   = 1;
    pr.theta.V   = 4;
    pr.theta.fun = @(t) normpdf(t, pr.theta.m, sqrt(pr.theta.V)); % normal
    
%     pr.theta.a   = 5;
%     pr.theta.b   = xopt(3)*(pr.theta.a-1);
%     pr.theta.fun = @(t) gampdf(1/t, pr.theta.a, 1/pr.theta.b); % IG

%% Estimation - MAP without prior
% Maximize the marginal likelihood to get bet, tau, theta, sigma

% initial guess
sig20  = sig2;
theta0 = theta;
tau0   = tau;

% objective function
para0 = [sig20;theta0;tau0];
% para0 = [0.05, 0.5234, 0.1846, 0.4846];
option.Display = 'iter';
objfun = @(para) objfun_reg(Y,X,para);
[x,fval] = fminsearch(objfun, para0,option);
xopt = x;

%% Estimation - MAP with prior
% Maximize the marginal likelihood to get bet, tau, theta, sigma

% initial guess
sig20  = sig2;
theta0 = theta;
tau0   = tau;

% objective function
para0 = [sig20;theta0;tau0];
% para0 = [0.05, 0.5234, 0.1846, 0.4846];
option.Display = 'iter';
objfun = @(para) objfun_reg(Y,X,para) ...
    -log(pr.sig2.fun(para(1))) ...
    -log(pr.theta.fun(para(2))) ...
    -log(pr.tau.fun(para(3)));
[x,fval] = fminsearch(objfun, para0,option);
xpost = x;


%% Gibbs sampling

% initialization of gibbs sampler (from the marginal likelihood opt)
sig2old  = xopt(1);
thetaold = xpost(end-1);
tauold   = xpost(end);

% matrix to store
mh_eta   = zeros(nsimul,nobs);
mh_sig2  = zeros(nsimul,1);

mh_theta = zeros(nsimul,1);
mh_thetarej = zeros(nsimul,1);

mh_tau   = zeros(nsimul,1);
mh_taurej = zeros(nsimul,1);


% -----------------------------------------------------------------------
% Gibbs sampler starts here
% -----------------------------------------------------------------------
tic;
for simulind = 1:1:nsimul

    % -----------------------------------------------------------------------
    % eta update - multivariate normal
    Cn       = gp_K(X,X,tauold,thetaold);
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
    % tau update - MH step
    
    % step 1: propose tau
    taunew = tauold + Ctau*randn(1,1);
    if taunew >0 %we only care positive tau
        % step 2: calculate MH ratio
        Cn_new   = gp_K(X,X,taunew,thetaold);
        post_new = mvnpdf(etaold', zeros(1,nobs), Cn_new)*pr.tau.fun(taunew);
        
        Cn_old   = gp_K(X,X,tauold,thetaold);
        post_old = mvnpdf(etaold', zeros(1,nobs), Cn_old)*pr.tau.fun(tauold);
    else
        % if new tau is less than zero, we reject
        post_new = -inf;
        post_old = inf;
    end
        
    % step 3: reject/accept
    r = log(post_new)-log(post_old);
    if rand < exp(r) % accept
        tauold = taunew;
        taurej = 0;
    else
        taurej = 1;
    end

    % -----------------------------------------------------------------------
    % theta update - MH step
    
    % step 1: propose theta
    thetanew = thetaold + Ctheta*randn(1,1);
    if thetanew >0 %we only accept thetanew >0
        % step 2: calculate MH ratio
        Cn_new   = gp_K(X,X,tauold,thetanew);
        post_new = mvnpdf(etaold', zeros(1,nobs), Cn_new)*pr.theta.fun(thetanew);
        
        Cn_old   = gp_K(X,X,tauold,thetaold);
        post_old = mvnpdf(etaold', zeros(1,nobs), Cn_old)*pr.theta.fun(thetaold);
    else % if new theta is less than zero, we reject
        post_new = -inf;
        post_old = inf;
    end
    % step 3: reject/accept
    r = log(post_new)-log(post_old);
    if rand < exp(r) % accept
        thetaold = thetanew;
        thetarej = 0;
    else
        thetarej = 1;
%         disp('thetarej');
    end
    
    
    % -----------------------------------------------------------------------
    % Store matrix
    mh_eta(simulind,:)    = etaold';
    mh_sig2(simulind,1)   = sig2old;
    
    mh_tau(simulind,:)    = tauold;
    mh_taurej(simulind,1) = taurej;
    
    mh_theta(simulind,:)    = thetaold;
    mh_thetarej(simulind,1) = thetarej;
    
    % Report
    if rem(simulind,100) == 0
        disp([num2str(simulind), ' / ', num2str(nsimul), ' th iteration']);
        disp(['rej theta : ', num2str(mean(mh_thetarej(1:simulind,:)))]);
        disp(['rej tau : ', num2str(mean(mh_taurej(1:simulind,:)))]);
    end
end
toc;

%% In-sample prediction at posterior mean

% at posterior mean 
theta1 = mean(mh_theta,1);
tau1   = mean(mh_tau,1);
sig21  = mean(mh_sig2,1);

Xstar = linspace(-5,5,100)'; %interpolation points

[Zm, ZV] = gp_pred(X, Y, Xstar, theta1, tau1, sig21);
Zup  = Zm + 1.96*sqrt(diag(ZV));
Zlow = Zm - 1.96*sqrt(diag(ZV));

% at arbitrary hyperparameter
theta2 = 1;
tau2   = 1;
sig22  = 1;
[Zm2, ZV2] = gp_pred(X, Y, Xstar, theta2, tau2, sig22);
Zup2  = Zm2 + 1.96*sqrt(diag(ZV2));
Zlow2 = Zm2 - 1.96*sqrt(diag(ZV2));


%% Plot 1: MCMC sampler
figure(1)
subplot(3,1,1)
plot(mh_tau)
set(gca, 'fontsize', 15, 'linewidth', 2);
title(['tau : rej. rate = ', num2str(mean(mh_taurej))], 'fontsize', 20);
subplot(3,1,2)
plot(mh_theta)
set(gca, 'fontsize', 15, 'linewidth', 2);
title(['theta : rej. rate = ', num2str(mean(mh_thetarej))], 'fontsize', 20);
subplot(3,1,3)
plot(mh_sig2)
title(['sig2'], 'fontsize', 20);
set(gca, 'fontsize', 15, 'linewidth', 2);

% estimates
tab = [{'sig2'; 'theta'; 'tau'}, num2cell([xopt, xpost, mean([mh_sig2, mh_theta, mh_tau])', std([mh_sig2, mh_theta, mh_tau])'])];
disp('estimtes: MLE, MAP, Posterior mean, std');
disp(tab);

%% Plot 2a: In-sample prediction at the posterior mode
figure(2)
% plot(Xstar,Zm, 'k', 'linewidth', 2)
hold on
plot(X,Y, 'k*', 'markersize', 10, 'linewidth', 1)
plot(Xstar,2*sin(Xstar), 'b--', 'linewidth', 2);
l = legend('Data', 'True');
set(l, 'fontsize', 15);
jbfill(Xstar',Zup',Zlow','r','r',1,0.4)
hold off
set(gca, 'linewidth', 2, 'fontsize', 20);
title('In-sample prediction band (95%) at the posterior mode', 'fontsize', 20);

%% Plot 2b: In-sample prediction at the posterior mode
figure(20)
% plot(Xstar,Zm, 'k', 'linewidth', 2)
hold on
plot(X,Y, 'k*', 'markersize', 10, 'linewidth', 1)
plot(Xstar,2*sin(Xstar), 'b--', 'linewidth', 2);
l = legend('Data', 'True');
set(l, 'fontsize', 15);
jbfill(Xstar',Zup2',Zlow2','r','r',1,0.4)
hold off
set(gca, 'linewidth', 2, 'fontsize', 20);
title('In-sample prediction band (95%) at the \theta=1, \tau=1, \sigma^{2}=1', 'fontsize', 20);


%% Plot 3: prior-posterior
% pp_theta = gamrnd(pr.theta.a, 1/pr.theta.b, 1000,1).^(-1);
pp_theta   = [];
while(size(pp_theta,1)<1000)
    abc = pr.theta.m + sqrt(pr.theta.V)*randn(1,1);
    if abc>0
        pp_theta = [pp_theta; abc];
    end
end


pp_tau   = [];
while(size(pp_tau,1)<1000)
    abc = pr.tau.m + sqrt(pr.tau.V)*randn(1,1);
    if abc>0
        pp_tau = [pp_tau; abc];
    end
end

ymax = max([pp_theta; mh_theta]);
ymin = min([pp_theta; mh_theta]);

xmax = max([pp_tau; mh_tau]);
xmin = min([pp_tau; mh_tau]);


figure(3);
subplot(1,2,1)
scatter(pp_tau, pp_theta,3);
xlim([xmin, xmax]);
ylim([ymin, ymax]);
title('Pior', 'fontsize', 20);
set(gca, 'linewidth', 2, 'fontsize', 15)
subplot(1,2,2)
scatter(mh_tau, mh_theta, 3, 'r');
set(gca, 'linewidth', 2, 'fontsize', 15)
xlim([xmin, xmax]);
ylim([ymin, ymax]);
title('Posterior', 'fontsize', 20);

figure(4);
x = pp_theta;
y = mh_theta([1:10:nsimul],1);

gl = min([x;y]);
gl = 0;
gh = max([x;y]);
grid = linspace(gl,gh,100)';
fx = ksdensity(x,grid);
fy = ksdensity(y,grid);

plot(grid, fx, 'b', 'linewidth', 3);
hold on
plot(grid, fy, 'r', 'linewidth', 3);
hold off
set(gca, 'linewidth', 2, 'fontsize', 15)
l = legend('Prior', 'Posterior');
set(l, 'fontsize', 15);
title('theta', 'fontsize', 20);

figure(5);
x = pp_tau;
y = mh_tau([1:10:nsimul],1);

gl = min([x;y]);
gl = 0;
gh = max([x;y]);
grid = linspace(gl,gh,100)';
fx = ksdensity(x,grid);
fy = ksdensity(y,grid);

plot(grid, fx, 'b', 'linewidth', 3);
hold on
plot(grid, fy, 'r', 'linewidth', 3);
hold off
l = legend('Prior', 'Posterior');
set(l, 'fontsize', 15);
set(gca, 'linewidth', 2, 'fontsize', 15)
title('tau', 'fontsize', 20);






