#Some Simple Examples to test the GP Regression Code
source("GP_regression_simple.R")

#Sine function
N <- 50
x <- runif(N, -5, 5)
y <- 2*sin(x) + rnorm(length(x))
x.star <- seq(from = -5, to = 5, by = 0.1)

results <- GP.reg(x, y, x.star, k.rbf, sigma.squared=1)
plot.GP(results)
points(x.star, 2*sin(x.star), col = 'red', type = 'l')

#Linear Function -- there are some boundary issues here
N <- 100
x <- runif(N, -5, 5)
y <- x + rnorm(length(x))
x.star <- seq(from = -5, to = 5, by = 0.1)

results <- GP.reg(x, y, x.star, k.rbf, sigma.squared=1)
plot.GP(results)
points(x.star, x.star, col = 'red', type = 'l')


#Quadratic Function -- again, some boundary issues since it tries to shrink to zero in regions where there are no observations
N <- 100
x <- runif(N, -5, 5)
y <- x^2 + rnorm(length(x))
x.star <- seq(from = -5, to = 5, by = 0.1)

results <- GP.reg(x, y, x.star, k.rbf, sigma.squared=1)
plot.GP(results)
points(x.star, x.star^2, col = 'red', type = 'l')


#See if it chokes on an example with many observations
N <- 1000
x <- runif(N, -5, 5)
y <- x + rnorm(length(x))
x.star <- seq(from = -5, to = 5, by = 0.1)

my.time <- system.time(results <- GP.reg(x, y, x.star, k.rbf, sigma.squared=1)) 
my.time


#Try tripling the previous experiment
#N <- 3000
#x <- runif(N, -5, 5)
#y <- x + rnorm(length(x))
#x.star <- seq(from = -5, to = 5, by = 0.1)

#my.time <- system.time(results <- GP.reg(x, y, x.star, k.rbf, sigma.squared=1)) 
#my.time  


#Really try pushing the envelope
#N <- 5000
#x <- runif(N, -5, 5)
#y <- x + rnorm(length(x))
#x.star <- seq(from = -5, to = 5, by = 0.1)

#my.time <- system.time(results <- GP.reg(x, y, x.star, k.rbf, sigma.squared=1)) 
