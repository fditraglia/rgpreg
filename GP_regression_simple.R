#GP_regression_simple.R
#July 9th, 2011
#Francis J. DiTraglia


#This file implements the simplest possible Gaussian Process Regression model, using the algorithm given on page 19 of Rassmussen & Williams' book "Gaussian Processes for Machine Learning" For simplicity, we start with regression using a single input variable.

#Some Preliminary notes:

  #Outer Product in R
    #outer(a, b, g)[i,j] is g(a[i], b[j])

  #Cholesky Factorization in R
    #R <- chol(M) #Returns upper triangular factor of Cholesky decomposition, i.e. the matrix R such that R'R = M for symmetric, positive definite matrix M

  #Solving Triangular Systems in R 
    #Let R be right (aka upper) triangular, L left (aka lower) triangular.
    #x <- backsolve (R, b) #solves R x = b
    #x <- forwardsolve(L, b) #solves L x = b
    #NB: both backsolve and forwardsolve allow us to put a MATRIX B in place of the vector b. In this case, each column of b corresponds to a RHS for a DIFFERENT linear system and the function returns a matrix whose columns are the respective solutions.



##########################################################
#     FUNCTION NAME: GP.reg
#
#     DESCRIPTION:  Given training inputs (x.train)
#                   output data (y), additive noise 
#                   level (sigma.squared), 
#                   covariance function (k), returns 
#                   the posterior mean function and 
#                   covariance matrix of a Gaussian 
#                   Process evaluated at test 
#                   covariate levels (x.test). We
#                   assume a prior mean of zero.
#
#     DETAILS:  y should be a vector of input data, 
#               x.train and x.test covariate vectors, 
#               sigma.squared a positive constant, and
#               k a covariance function taking two 
#               scalar arguments. Note that this
#               function only handles regression with a
#               single input.
#
##########################################################
GP.reg <- function(x.train, y, x.test, k, sigma.squared){
  
  #For simplicity denote training inputs by x and the test inputs by z
  x <- x.train
  z <- x.test
  
  #Form the covariance matrices for test and training points
  K.xx <- outer(x, x, k)
  K.xz <- outer(x, z, k)
  K.zz <- outer(z, z, k)
  K.zx <- t(K.xz)
  
  #Take Cholesky decomposition of the "BIG" matrix we want to avoid inverting
  A <- K.xx + diag(sigma.squared, length(x))
  R <- chol(A)
  #Note that L from the book's notation is t(R). We don't want to store both L and R because these are HUGE matrices
  
  #Calculate Posterior Mean
  b <- backsolve(R, forwardsolve(t(R), y)) #Solve Ab = y efficiently in two steps
  post.mean <- K.zx %*% b
  
  #Posterior Variance Matrix
  C <- forwardsolve(t(R), K.xz)
  post.var <- K.zz - t(C) %*% C
  
  #Return list object containing posterior mean vector, covariance matrix and all input parameters and data
  out <- list(post.mean = post.mean, post.var = post.var, x.train = x.train, y=y, x.test=x.test, sigma.squared=sigma.squared)
  return(out)
  
  
}#END GP.reg
##########################################################






##########################################################
#     FUNCTION NAME: plot.GP
#
#     DESCRIPTION:  Plot method for one-dimensional GP 
#                   regression.
#
#     DETAILS:  Given input list of the form returned by
#               the function GP.reg, plots training
#               data, posterior mean evaluated at test
#               points, and pointwise 2 std. dev. error
#               posterior error bands.
##########################################################
plot.GP <- function(GP.reg.results){
  
  #Extract results for plotting
  post.mean <- GP.reg.results$post.mean
  post.var <- GP.reg.results$post.var
  x.train <- GP.reg.results$x.train
  y <- GP.reg.results$y
  x.test <- GP.reg.results$x.test
  
  #Calculate (+/-) 2 standard deviation posterior error bands
  own.sd <- sqrt(diag(post.var))
  upper <- post.mean + 2*own.sd
  lower <- post.mean - 2*own.sd
  
  #Find minimum and maximum y-axis positions for plotting
  y.min <- min(min(y), min(lower))
  y.max <- max(max(y), max(upper))
  
  #Find minimum and maximum x-axis positions for plotting
  x.min <- min(min(x.test), min(x.train))
  x.max <- max(max(x.test), max(x.train))

  #Set up an empty plot with the appropriate x and y limits
  plot(NULL, col = 'white', main = 'Simple Gaussian Process Regression', xlim = c(min(x.test), max(x.test)), ylim = c(y.min, y.max), xlab = 'x', ylab = 'y')

  #Define and plot a polygon representing the error bands
  xx <- c(x.test, rev(x.test))
  yy <- c(lower, rev(upper))
  polygon(xx, yy, border = NULL, col = 'gray')

  #Plot the training data
  points(x.train, y, pch = 3, col = 'red')
  
  #Plot the posterior
  points(x.test, post.mean, type = 'l', col = 'blue')  
    
}#END plot.GP
##########################################################



########################################################## 
#Covariance function for radial basis function, aka squared exponential, when x.1, and x.2 are scalars.
k.rbf <- function(x.1, x.2, ell = 1){
  
  covar <- exp(-1/(2 * ell^2) * (x.1 - x.2)^2)
  
  return(covar)

}#END k.rbf
##########################################################
  




