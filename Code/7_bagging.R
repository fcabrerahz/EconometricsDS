# install.packages(c("wooldridge","rpart"))  # if needed
library(wooldridge)
library(rpart)

set.seed(123)

# Data & split
dat <- na.omit(wage1[, c("wage","educ","exper","tenure")])
n   <- nrow(dat)
id  <- sample(seq_len(n), floor(0.7*n))
train  <- dat[id, ]
test  <- dat[-id, ]

# ==========================
# Manual bagging
# ==========================

B <- 200
pred_mat <- matrix(NA_real_, nrow = nrow(test), ncol = B)

ctrl <- rpart.control(minsplit = 20, cp = 0)  # grow sin penalización ("C" = 0) (unpruned)

for (b in 1:B) {
  idx_b <- sample(seq_len(nrow(train)), replace = TRUE)  #vector of row numbers from 1 to n, with replacement (some obs repeated some never sampled!)
  tb    <- train[idx_b, ] # Creates the bootstrap training dataset tb. if tr has 700 rows, then tb also has 700 rows (*some duplicates*).
  fit_b <- rpart(wage ~ educ + exper + tenure, data = tb, method = "anova", control = ctrl)
  pred_mat[, b] <- predict(fit_b, newdata = test) #Uses the bth tree to predict on the test data.
}

# Aggregate (bagged prediction = average across bootstrap trees)
pred_bag <- rowMeans(pred_mat)
rmse_bag <- sqrt(mean((test$wage - pred_bag)^2))
cat("Bagging (manual rpart) | Test RMSE:", round(rmse_bag, 3), "\n")

# ==========================
# B) Bagging via randomForest (mtry = p)
# ==========================
p <- ncol(train) - 1  # number of predictors (educ, exper, tenure)

rf_bag <- randomForest(
  wage ~ educ + exper + tenure,
  data   = train,
  ntree  = 500,        # number of trees
  mtry   = p,          # use ALL predictors at each split -> pure bagging
  nodesize = 5,        # min obs in terminal nodes
  importance = TRUE,
  keep.inbag = TRUE
)

# Out-of-bag (OOB) RMSE and Test RMSE
rmse <- function(y, yhat) {
  sqrt(mean((y - yhat)^2))
}

oob_rmse <- sqrt(tail(rf_bag$mse, 1))
pred_te_rf <- predict(rf_bag, newdata = test)
rmse_rf    <- rmse(test$wage, pred_te_rf)


cat("RF bagging (mtry=p)   | OOB RMSE:", round(oob_rmse, 3),
    "| Test RMSE:", round(rmse_rf, 3), "\n",)

# ==========================
# C) Examples: prediction, accuracy, importance
# ==========================

# Predict for an hypothetical profile
new_person <- data.frame(educ = 16, exper = 5, tenure = 2)
pred_rf_new     <- predict(rf_bag, newdata = new_person)

cat("Pred (RF bagging) new person:", round(pred_rf_new, 3), "\n")

# Variable importance (RF)
print(importance(rf_bag)) #how much could the MSE increase if we delete x var.

# Plot:
varImpPlot(rf_bag, main = "Variable Importance (Bagging with mtry=p)")
##IncNodePurity (importance in spliting) tends to favor variables with many possible split points, e.g., continuous or high-cardinality variables.
## This is why many practitioners prefer %IncMSE, which is more reliable.

# Quick predicted vs actual (RF) diagnostic
plot(test$wage, pred_te_rf, pch = 19, col = rgb(0,0,1,0.4),
      xlab = "Actual wage", ylab = "Predicted wage (RF bagging)")
 abline(0, 1, col = "red", lwd = 2)


