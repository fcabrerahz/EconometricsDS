#  Data: create near-collinearity 
set.seed(123)
n <- 200
x1 <- rnorm(n)
x2 <- x1 + rnorm(n, sd = 1e-3)          # near collinearity
X  <- cbind(1, x1, x2)
beta_true <- c(0.5, 1, -1)
y  <- X %*% beta_true + rnorm(n, sd = 0.1)

XtX <- t(X) %*% X
Xty <- t(X) %*% y

# OLS
beta_ols <- solve(XtX, Xty)
yhat_ols <- X %*% beta_ols
rmse_ols <- sqrt(mean((y - yhat_ols)^2))
cat("OLS coefficients:\n"); print(drop(beta_ols))
cat("OLS RMSE:", rmse_ols, "\n\n")

# Penalty matrix (don't penalize intercept)
p <- ncol(X)
P <- diag(p); P[1,1] <- 0

# Ridge solver (matrix form)
ridge_fit <- function(X, y, lambda, P) {
  solve(t(X) %*% X + lambda * P, t(X) %*% y)
}

lambda <- 1e-2
beta_ridge <- ridge_fit(X, y, lambda, P)
yhat_ridge <- X %*% beta_ridge
rmse_ridge <- sqrt(mean((y - yhat_ridge)^2))
cat("Ridge (lambda =", lambda, ") coefficients:\n"); print(drop(beta_ridge))
cat("Ridge RMSE:", rmse_ridge, "\n\n")

# --- Variance–covariance of ridge (homoskedastic) ---
sigma2_hat <- mean((y - yhat_ridge)^2)  # simple estimate of σ^2

Ainv <- solve(XtX + lambda * P)
Var_beta <- sigma2_hat * Ainv %*% XtX %*% Ainv
se_beta  <- sqrt(diag(Var_beta))

cat("Ridge coefficients:\n"); print(drop(beta_ridge))
cat("\nStandard errors (manual formula):\n"); print(se_beta)

## RIDGE REGRESSION command!

# Install
install.packages("glmnet")
library(glmnet)

# Simulate data
set.seed(123)
n <- 200
p <- 10
x <- matrix(rnorm(n * p), n, p)
y <- 3 + x[,1]*2 - x[,2]*1.5 + rnorm(n)

# Ridge regression (alpha = 0, makes it Ridge)
ridge_mod <- glmnet(x, y, alpha = 0)

# Show coefficients at different lambda values
print(ridge_mod)

#visualize

plot(ridge_mod, xvar = "lambda", label = TRUE)


###############
#Use cv.glmnet() to automatically find the lambda that minimizes prediction error:

cv_ridge <- cv.glmnet(x, y, alpha = 0)

# Best lambda values
cv_ridge$lambda.min     # Lambda with minimum CV error

# Coefficients at best lambda
coef(cv_ridge, s = "lambda.min")

##Variance should be computed manually.



