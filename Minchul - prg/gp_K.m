% Covariance kernel for GP
% Current version uses squared-exponential cov
function C = gp_K(x1, x2, tau, theta)

% kernel function
fun_se = @(x,y) tau*exp(-1/theta*(x-y)^2); %SE kernel

% get covariance function
n1 = size(x1,1);
n2 = size(x2,1);
C = zeros(n1,n2);
for i=1:1:n1
    for j=1:1:n2
        C(i,j) = fun_se(x1(i),x2(j));
    end
end
if n1 == n2
C = C + 1e-10*eye(n1); % without this C can be non PSD...
end
% chol(Cn);

% tic
% [xx1, xx2] = meshgrid(x1,x2);
% C2 = arrayfun(fun_se,xx1,xx2);
% C2 = C2';
% toc
% C - C2