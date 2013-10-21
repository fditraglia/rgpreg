#GP_regression.R
#July 11th, 2011
#Francis J. DiTraglia


#This file builds a Gibbs sampler building on the code from the script GP_regression_simple.R and adding a prior for sigma.squared to the GP regression modes described on page 19 of Rassmussen & Williams' book "Gaussian Processes for Machine Learning" For simplicity, we start with regression using a single input variable.



##########################################################
#     FUNCTION NAME: Inverse.Gamma
#
#     DESCRIPTION:  Generates single variate from Inverse
#                   Gamma Distribution for use in Gibbs 
#                   sampler
#
#     DETAILS:  a is the shape parameter, d the rate
#               parameter, as defined in Greenberg (2008)
#               "Introduction to Bayesian Econometrics"
##########################################################
Inverse.Gamma <- function(a, d){
  
  gamma.sim <- rgamma(n = 1, shape = a, rate = d)
  inverse.gamma.sim <- 1/gamma.sim
  
  return(inverse.gamma.sim)
  
}#END Inverse.Gamma
##########################################################

  
  
  
##########################################################
#     FUNCTION NAME: MV.normal
#
#     DESCRIPTION:  Generates single variate from
#                   Multivariate Normal distribution for
#                   use in Gibbs Sampling
#
#     DETAILS:  m is the mean vector, S the covariance
#               matrix. S must be a positive definite
#               square symmetric matrix of the same 
#               dimension as m.
##########################################################
MV.Normal <- function(m, S){
  
  d <- length(m)
  
  #Draw d standard normal variates
  z <- rnorm(n=d)
  
  #Take the Cholesky Factorization (Matrix Square Root) of S
  R <- chol(S) #Using more standard notation L is t(R)
  
  x <- m + t(R) %*% z

  return(x)
  
}#END MV.Normal
##########################################################

  
  
  
##########################################################
#     FUNCTION NAME: k.RBF
#
#     DESCRIPTION:  Covariance function for radial basis 
#                   function, aka squared exponential, 
#                   over scalar inputs
#
#     DETAILS:  x.1 and x.2 are two scalars and ell > 0
#               is the characteristic length-scale as
#               given in Rassmussen & William's book.
##########################################################
k.RBF <- function(x.1, x.2, ell = 1){
  
  covar <- exp(-1/(2 * ell^2) * (x.1 - x.2)^2)
  
  return(covar)

}#END k.rbf
##########################################################

  
  
  
##########################################################
#     FUNCTION NAME: GP.posterior
#
#     DESCRIPTION:  
#
#     DETAILS:  
##########################################################
GP.reg <- function(K.xx, K.zz){
  
  #For simplicity denote training inputs by x and the test inputs by z
  x <- x.train
  z <- x.test
  
  #Form the covariance matrices for test and training points
  K.xx <- outer(x, x, k)
  K.xz <- outer(x, z, k)
  K.zz <- outer(z, z, k)
  
  #Take Cholesky decomposition of the "BIG" matrix we want to avoid inverting
  A <- K.xx + diag(sigma.squared, length(x))
  R <- chol(A)
  #Note that L from the book's notation is t(R). We don't want to store both L and R because these are HUGE matrices
  
  #Calculate Posterior Mean
  b <- backsolve(R, forwardsolve(t(R), y)) #Solve Ab = y efficiently in two steps
  post.mean <- t(K.xz) %*% b
  
  #Posterior Variance Matrix
  C <- forwardsolve(t(R), K.xz)
  post.var <- K.zz - t(C) %*% C
  
  #Return list object containing posterior mean vector, covariance matrix and all input parameters and data
  out <- list(post.mean = post.mean, post.var = post.var)
  return(out)
  
  
}#END GP.reg
##########################################################



##########################################################
#     FUNCTION NAME: gibbs.GP.reg
#
#     DESCRIPTION:  Generates posterior samples for
#                   Bayesian univariate Nonparametric 
#                   regression via the Gibbs sampler. The
#                   regression function is given a zero
#                   mean Gaussian Process prior, the noise
#                   term is assumed additive and normal
#                   with unknown variance parameter given
#                   an inverse gamma distribution, 
#                   independent of the Gaussian Process.
#
#     DETAILS:  
##########################################################
gibbs.GP.reg <- function(x, y, x.new, k, a0, d0, sigma.squared.start, N.reps){

  #To get conditional conjugacy for the posterior distribution of sigma.squared, we need to draw samples from the posterior of f at both the test points AND training points. Our convention will be for the first length(x) of the "test points" z to be the training points and the remaining elements to be the test points actually specified by the user, namely x.new. We will call z the "augmented" test points. 
  z <- c(x, x.new)
  
  #Number of observations
  n <- legnth(y)
  
  #Initialize matrix to store samples from posterior of f evaluated at z, the "augmented" test points. Each column is a sample from the posterior while each row is the value of f evaluated at a particular point, in parallel with the entires of z.
  f <- matrix(NA, nrow = length(z), ncol = N.reps)
  
  #Initialize vector to store samples from posterior of sigma.squared
  sigma.squared <- rep(NA, N.reps)
  
  
  #Hmmmm, don't need all three of these given my augmented test vector
  #Caclulate all quantities that don't change in the loop
    #covariance matrices for training and "augmented test" points
  K.xx <- outer(x, x, k)
  K.xz <- outer(x, z, k) #K.zx is just t(K.xz) but we don't want to store it
  K.zz <- outer(z, z, k) 
    #posterior parameter a1 for sigma.squared
  a1 <- a0 + n 
  
  
  #Draw first sample for f and sigma.squared outside of for loop
  
  for(i in 2:N.reps){
    
    
    
    
  }#END for
  
  
}#END gibbs.GP.reg
##########################################################
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
