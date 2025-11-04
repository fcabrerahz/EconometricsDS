# Install once (if needed)
# install.packages("glmnet")

library(glmnet)

# --- Example data (you can replace with your X, y) ---
set.seed(123)
n <- 200
x1 <- rnorm(n)
x2 <- x1 + rnorm(n, sd = 1e-3)          # collinearity
X  <- cbind(x1, x2)
y  <- 0.5 + 1*x1 - 1*x2 + rnorm(n, sd = 0.1)

# glmnet expects a numeric matrix for x and a numeric vector for y
x_mat <- as.matrix(X)
y_vec <- as.numeric(y)

# --- LASSO with cross-validation (automatic λ tuning) ---
# alpha=1 => LASSO; family="gaussian" for regression
cvfit <- cv.glmnet(x = x_mat, y = y_vec, alpha = 1, family = "gaussian",
                   nfolds = 10, standardize = TRUE, intercept = TRUE)

# Best lambdas
lambda_min <- cvfit$lambda.min   # minimizes CV error
lambda_1se <- cvfit$lambda.1se   # 1-SE rule (more shrinkage)

cat("lambda.min =", signif(lambda_min, 6), "\n")
cat("lambda.1se =", signif(lambda_1se, 6), "\n")

# Coefficients at the chosen λ
coef_min  <- coef(cvfit, s = "lambda.min")
coef_1se  <- coef(cvfit, s = "lambda.1se")

cat("\nNonzero coefficients at lambda.min:\n")
print(coef_min[coef_min != 0, , drop = FALSE])

# Predictions (e.g., in-sample; replace newx with test data as needed)
yhat_min <- predict(cvfit, newx = x_mat, s = "lambda.min")
rmse_min <- sqrt(mean((y_vec - yhat_min)^2))
cat("\nRMSE at lambda.min:", rmse_min, "\n")
