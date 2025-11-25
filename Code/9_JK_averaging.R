###############################################################
# Ridge, Tree, Random Forest
# Jackknife (LOO) Selection and Jackknife Model Averaging (JMA)
# Using wage1 from wooldridge
###############################################################

library(wooldridge)
library(glmnet)
library(rpart)
library(randomForest)
library(quadprog)

set.seed(123)

# 1. Load data ---------------------------------------------------------
data("wage1")
dat <- subset(wage1, !is.na(wage))

# Use three regressors that definitely exist
X <- as.matrix(dat[, c("educ", "exper", "tenure")])
y <- dat$wage
n <- length(y)

# 2. Fit models on full sample ----------------------------------------

# Ridge regression with CV-chosen lambda
cv_ridge   <- cv.glmnet(X, y, alpha = 0)
ridge_full <- glmnet(X, y, alpha = 0, lambda = cv_ridge$lambda.min)

# Regression tree
tree_full <- rpart(wage ~ educ + exper + tenure,
                   data = dat, method = "anova")

# Random forest
rf_full <- randomForest(wage ~ educ + exper + tenure,
                        data = dat)

# 3. Leave-one-out predictions ----------------------------------------

M <- 3   # Ridge, Tree, RF
model_names <- c("Ridge", "Tree", "RandomForest")
pred_loo <- matrix(NA, n, M)
colnames(pred_loo) <- model_names

#removes observation i and estimates n times.
for (i in 1:n) {
  idx <- setdiff(1:n, i) #all indices except i.
  
  X_train   <- X[idx, , drop = FALSE]
  y_train   <- y[idx]
  dat_train <- dat[idx, ] #the full data frame without row i
  
  # Ridge
  ridge_fit <- glmnet(X_train, y_train, alpha = 0,
                      lambda = cv_ridge$lambda.min)
  pred_loo[i, "Ridge"] <- as.numeric(
    predict(ridge_fit, newx = matrix(X[i, ], nrow = 1))
  )
  
  # Tree
  tree_fit <- rpart(wage ~ educ + exper + tenure,
                    data = dat_train, method = "anova")
  pred_loo[i, "Tree"] <- predict(tree_fit, newdata = dat[i, ])
  
  # Random forest
  rf_fit <- randomForest(wage ~ educ + exper + tenure,
                         data = dat_train)
  pred_loo[i, "RandomForest"] <- predict(rf_fit,
                                         newdata = dat[i, ])
}

# 4. Jackknife errors and selection -----------------------------------

Etilde    <- y - pred_loo
cv_scores <- colSums(Etilde^2)

cv_scores
best_model <- model_names[which.min(cv_scores)]
cat("Jackknife-selected model:", best_model, "\n")

# 5. Jackknife model averaging (quadratic program) --------------------

Dmat <- t(Etilde) %*% Etilde
Dmat <- (Dmat + t(Dmat)) / 2
dvec <- rep(0, M)

Amat <- cbind(rep(1, M), diag(M))  # sum w = 1; w >= 0
bvec <- c(1, rep(0, M))
meq  <- 1

sol   <- solve.QP(Dmat, dvec, Amat, bvec, meq)
w_jma <- sol$solution
names(w_jma) <- model_names

cat("Jackknife averaging weights:\n")
print(round(w_jma, 4))

# 6. Compare JMA vs best single model ---------------------------------

JMA_CV <- sum((Etilde %*% w_jma)^2)
cat("CV(best model):", round(min(cv_scores), 4), "\n")
cat("CV(JMA):        ", round(JMA_CV, 4), "\n")

# 7. Predictions using full-sample models + JMA -----------------------

# Example: new data
newdata <- data.frame(
  educ   = c(12, 16, 10),
  exper  = c(5, 2, 20),
  tenure = c(2, 1, 10)
)

# Ridge predictions
pred_ridge_new <- as.numeric(
  predict(ridge_full, newx = as.matrix(newdata))
)

# Tree predictions
pred_tree_new <- as.numeric(
  predict(tree_full, newdata = newdata)
)

# Random forest predictions
pred_rf_new <- as.numeric(
  predict(rf_full, newdata = newdata)
)

# Stack predictions
pred_mat_new <- cbind(pred_ridge_new,
                      pred_tree_new,
                      pred_rf_new)
colnames(pred_mat_new) <- model_names

# JMA prediction
pred_jma_new <- as.numeric(pred_mat_new %*% w_jma)

# Results
results_new <- cbind(newdata,
                     Ridge        = pred_ridge_new,
                     Tree         = pred_tree_new,
                     RandomForest = pred_rf_new,
                     JMA          = pred_jma_new)

print(results_new)


