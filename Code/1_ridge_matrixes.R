## Near-singular X'X  → exploding OLS; then stabilize with ridge
set.seed(1)

n <- 200
# Two almost-identical predictors (rho ~ 0.999)
x1 <- rnorm(n)
x2 <- x1 + rnorm(n, sd = 1e-3)

# Design matrix with intercept
X <- cbind(1, x1, x2)                # n x p
p <- ncol(X)

# True beta (modest size) and outcome
beta_true <- c(0.5, 1, -1)           # intercept, x1, x2
y <- X %*% beta_true + rnorm(n, sd = 0.1)

# --- OLS by matrices ----------------------------------------------------------
XtX <- t(X) %*% X
Xty <- t(X) %*% y

# OLS (may be numerically unstable when XtX is ill-conditioned)
beta_ols <- solve(XtX, Xty)
cat("beta_ols:\n"); print(drop(beta_ols))

eigs <- eigen(XtX)$values
cat("Eigenvalues of X'X:\n"); print(signif(eigs, 6))
cat("Condition number kappa(X'X):\n"); print(kappa(XtX))

# Tiny perturbation of y -> check sensitivity of OLS
y_pert <- y; y_pert[1] <- y_pert[1] + 1e-2
beta_ols_pert <- solve(XtX, t(X) %*% y_pert)

cat("||beta_ols - beta_ols_pert||_2 (tiny change in y):\n")
print(norm(beta_ols - beta_ols_pert, type = "2"))

# --- Ridge regularization by matrices ----------------------------------------
# Choose lambda; do NOT penalize intercept (common practice)
lambda <- 1e-2
I <- diag(p); P[1, 1] <- 0           # no penalty on intercept

beta_ridge      <- solve(XtX + lambda * I, Xty)
beta_ridge_pert <- solve(XtX + lambda * I, t(X) %*% y_pert)

cat("beta_ridge (lambda =", lambda, "):\n"); print(drop(beta_ridge))

cat("||beta_ridge - beta_ridge_pert||_2 (same tiny change in y):\n")
print(norm(beta_ridge - beta_ridge_pert, type = "2"))

# Compare magnitudes
cat("L2-norms  |beta_ols|, |beta_ridge|:\n")
print(c(ols = norm(beta_ols, "2"), ridge = norm(beta_ridge, "2")))
