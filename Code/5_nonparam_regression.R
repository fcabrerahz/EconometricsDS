# Kernel (NW) and Local Linear Regression in R
# with Linear Regression Comparison
# - Data: Wooldridge's wage1 (from wooldridge package)
# - Gaussian kernel
# - Bandwidths: Rule-of-Thumb (ROT) and Cross-Validation (CV)

# install.packages(c("wooldridge","np","KernSmooth","ggplot2"))

library(wooldridge)
library(np)
library(KernSmooth)
library(ggplot2)

# 1) Load data
data("wage1", package = "wooldridge")

# 2) Define variables
dat <- subset(wage1, !is.na(wage) & !is.na(exper))
x <- as.numeric(dat$exper)
y <- as.numeric(dat$wage)
n <- length(y)

# Evaluation grid for smooths
xgrid <- seq(min(x), max(x), length.out = 300)

# ----------------------------
# A) Nadaraya–Watson (local constant)
# ----------------------------

# ROT bandwidth (Silverman’s rule)
h_rot_nw <- 1.06 * sd(x) * n^(-1/5)

# Fit NW (ROT)
nw_rot_fit <- ksmooth(x, y, kernel = "normal", bandwidth = h_rot_nw, x.points = xgrid)
nw_rot <- data.frame(x = nw_rot_fit$x, mhat = nw_rot_fit$y, method = "NW (ROT)")

# CV bandwidth (npregbw)
bw_nw_cv <- npregbw(ydat = y, xdat = x, regtype = "lc", ckertype = "gaussian")
nw_cv_model <- npreg(bws = bw_nw_cv, tydat = y, txdat = x, exdat = xgrid,
                     regtype = "lc", ckertype = "gaussian")
nw_cv <- data.frame(x = xgrid, mhat = as.numeric(nw_cv_model$mean), method = "NW (CV)")

# ----------------------------
# B) Local Linear Regression
# ----------------------------

# ROT bandwidth via plug-in (dpill)
h_rot_ll <- dpill(x, y)
ll_rot_fit <- locpoly(x, y, degree = 1, bandwidth = h_rot_ll, kernel = "normal",
                      gridsize = length(xgrid), range.x = range(x))
ll_rot <- data.frame(x = ll_rot_fit$x, mhat = ll_rot_fit$y, method = "Local Linear (ROT)")

# CV bandwidth via npregbw
bw_ll_cv <- npregbw(ydat = y, xdat = x, regtype = "ll", ckertype = "gaussian")
ll_cv_model <- npreg(bws = bw_ll_cv, tydat = y, txdat = x, exdat = xgrid,
                     regtype = "ll", ckertype = "gaussian")
ll_cv <- data.frame(x = xgrid, mhat = as.numeric(ll_cv_model$mean), method = "Local Linear (CV)")

# ----------------------------
# C) Parametric Linear Regression (OLS)
# ----------------------------
ols_model <- lm(y ~ x)
ols_pred <- data.frame(x = xgrid,
                       mhat = predict(ols_model, newdata = data.frame(x = xgrid)),
                       method = "Linear Regression (OLS)")

# ----------------------------
# Combine all results
# ----------------------------
curves <- rbind(nw_rot, nw_cv, ll_rot, ll_cv, ols_pred)

# ----------------------------
# Plot: Scatter + All Smooths
# ----------------------------
ggplot(data.frame(x = x, y = y), aes(x, y)) +
  geom_point(alpha = 0.35, size = 1.5) +
  geom_line(data = curves, aes(x = x, y = mhat, color = method, linetype = method), linewidth = 1) +
  labs(title = "Kernel (NW) and Local Linear Regression vs OLS",
       subtitle = "Gaussian kernel — Bandwidths: Rule-of-Thumb (ROT) and Cross-Validation (CV)",
       x = "Experience (years)", y = "Wage",
       color = "Estimator", linetype = "Estimator") +
  theme_minimal(base_size = 12)

# ----------------------------
# Report bandwidths used
# ----------------------------
cat("\n--- Bandwidths ---\n")
cat("NW ROT bandwidth (Silverman):         ", signif(h_rot_nw, 6), "\n")
cat("NW CV bandwidth (npregbw):            ", signif(bw_nw_cv$bw, 6), "\n")
cat("Local Linear ROT bandwidth (dpill):   ", signif(h_rot_ll, 6), "\n")
cat("Local Linear CV bandwidth (npregbw):  ", signif(bw_ll_cv$bw, 6), "\n")


