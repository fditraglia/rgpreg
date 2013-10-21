#Extremely Simple Gibbs Sampler Example
#Francis DiTraglia
#July 10th, 2011

#Linear Regression with one covariate and independent normal-inverse gamma prior in which we know that the intercept term is zero

#Here b is the slope coefficient, sigma.squared the error variance
#b0 is the prior mean of b, B0 the prior variance 

#Notes: 

#R lacks a inverse gamma distribution. If  x ~ Gamma(a,b) and y = 1/x, then y has an Inverse Gamma(a,b) distribution

#R parameterizes the gamma distribution according to:
#f(x)= 1/(s^a Gamma(a)) x^(a-1) e^-(x/s)
#where a is the shape parameter, and s the scale parameter

#In contrast, Greenberg (2008) parameterizes it according to
#(b^a)/(Gamma(a)) x^(a-1) e^-(bx)
#where a is the shape parameter and b the rate parameter

#Note that s = 1/b. In fact, R gives the option to express the distribution in terms of b by setting rate = b (i.e. R knows that scale = 1/rate).


gibbs.reg <- function(x, y, b0=0, B0=100, a0=1, d0=1, sigma.squared.start, N.reps=1000){
  
  #Initialize b and sigma.squared
  b <- rep(NA, N.reps)
  sigma.squared <- rep(NA, N.reps)
  
  #Calculate any quantities not involving sigma.squared and b only once!
  B0.inv <- (1/B0)
  xx <- sum(x^2)
  xy <- sum(x*y)
  A <- B0.inv * b0
  n <- length(y)  #number of observations
  
  #Pump-priming step
  #Conditional Posterior variance for b
  B1 <- 1/(B0.inv + xx/sigma.squared.start)
  #Conditional Posterior mean for b
  b.bar <- B1 * (A + xy/sigma.squared.start)
  #Simulate from conditional posterior for b
  b[1] <- rnorm(n = 1, mean = b.bar, sd = sqrt(B1))
  
  #Shape parameter for conditional posterior of sigma squared
  a1 <- a0 + n
  #Rate parameter for conditional posterior of sigma squared
  d1 <- d0 + sum((y-b[1]*x)^2) 
  #Simulate from conditional posterior for sigma squared
  sigma.squared[1] <- 1/rgamma(n = 1, shape = a1/2, rate = d1/2)
  
  
  for(i in 2:N.reps){
    
    #Conditional Posterior variance for b
    B1 <- 1/(B0.inv + xx/sigma.squared[i-1])
    #Conditional Posterior mean for b
    b.bar <- B1 * (A + xy/sigma.squared[i-1])
    #Simulate from conditional posterior for b
    b[i] <- rnorm(n = 1, mean = b.bar, sd = sqrt(B1))
  
    #Shape parameter for conditional posterior of sigma squared
    a1 <- a0 + n
    #Rate parameter for conditional posterior of sigma squared
    d1 <- d0 + sum((y-b[i]*x)^2) 
    #Simulate from conditional posterior for sigma squared
    sigma.squared[i] <- 1/rgamma(n = 1, shape = a1/2, rate = d1/2)
 
    
  }#END for 
  
  out <- list(sigma.squared=sigma.squared, b=b)
  return(out)
  
  
}#END gibbs.reg


#Plot method for gibbs.reg
plot.gibbs <- function(results){
  
  error.variance <- results$sigma.squared
  reg.coeff <- results$b
  
  var.max <- max(error.variance)
  var.min <- min(error.variance)
  
  coeff.max <- max(reg.coeff)
  coeff.min <- min(reg.coeff)
  
  
  plot(error.variance[1], reg.coeff[1], main='Gibbs Samples', col = 'red', type = 'p', xlab=substitute(sigma^2), ylab = substitute(beta), xlim = c(var.min, var.max), ylim = c(coeff.min, coeff.max))
  
  for(i in 2:length(error.variance)){
  
  x1 <- error.variance[i-1]
  x2 <- error.variance[i]
  
  y1 <- reg.coeff[i-1]
  y2 <- reg.coeff[i]
  
  lines(x=c(x1, x2), y=c(y1, y1))
  
  lines(x=c(x2, x2), y=c(y1, y2))
    
  }#END for
  
  
}#END plot.gibbs  

  
#See how well it works
N <- 1000
x <- runif(N)
y <- x + rnorm(N)

sims <- gibbs.reg(x, y, N.reps = 1000, sigma.squared.start = 2)
plot.gibbs(sims)

Burnin <- 500
b <- sims$b[-(1:Burnin)]
s2 <- sims$sigma.squared[-(1:Burnin)]
  
plot(sims$b, type = 'l')
plot(sims$sigma.squared, type = 'l')

plot(density(b))
plot(density(s2))

mean(b)
median(b)

lm(y~x-1)



