# install.packages(c("wooldridge","rpart","rpart.plot"))

library(wooldridge)
library(rpart)
library(rpart.plot)

# Data
dat <- na.omit(wage1[, c("wage","educ","exper","tenure")])

# Fit regression tree (CART) with 10-fold CV (default in rpart)
set.seed(123)
tree_fit <- rpart(
  wage ~ educ + exper + tenure,
  data = dat,
  method = "anova",
  control = rpart.control(minsplit = 20, cp = 0.001)  
)

# Inspect CP table (training + cross-validated errors)
printcp(tree_fit)

# Choose the cp ("C" in the slides) that minimizes cross-validated error (xerror)
best_cp <- tree_fit$cptable[which.min(tree_fit$cptable[,"xerror"]), "CP"]

# Prune to optimal size
tree_pruned <- prune(tree_fit, cp = best_cp)

# Plot the pruned tree
rpart.plot(tree_pruned, type = 2, extra = 101, fallen.leaves = TRUE,
           main = "Regression Tree: wage ~ educ + exper + tenure")


