############################################################
# Bagging vs Random Forest on wage1
# Variable Importance + Prediction Comparison
############################################################

library(wooldridge)
library(randomForest)   # contains both bagging and RF
library(MASS)            # for true bagging alternative (if needed)

set.seed(123)

# 1. Load and prepare data ---------------------------------
data("wage1")
dat <- subset(wage1, !is.na(wage))

# Convert categorical variables to factors
dat$female   <- factor(dat$female)
dat$married  <- factor(dat$married)
dat$nonwhite <- factor(dat$nonwhite)

# Keep variables
wage_dat <- dat[, c("wage", "educ", "exper", "tenure",
                    "female", "married", "nonwhite")]

# 2. BAGGING ------------------------------------------------
# Bagging is Random Forest with mtry = p (all variables tried at every split)

p <- 6   # number of predictors

bag_mod <- randomForest(
  wage ~ educ + exper + tenure + female + married + nonwhite,
  data       = wage_dat,
  mtry       = p,        # ALL variables used: Bagging
  ntree      = 500,
  importance = TRUE
)

cat("\nBagging Model Summary:\n")
print(bag_mod)

cat("\nBagging Variable Importance:\n")
print(round(importance(bag_mod), 3)) 
#%IncMSE measures how much prediction error increases 
#when a variable’s information is removed. Uses OOB for MSE.

# 3. RANDOM FOREST -----------------------------------------

rf_mod <- randomForest(
  wage ~ educ + exper + tenure + female + married + nonwhite,
  data       = wage_dat,
  mtry       = 3,        # sqrt(p) or p/3; default for regression
  ntree      = 500,
  importance = TRUE
)

cat("\nRandom Forest Summary:\n")
print(rf_mod)

cat("\nRandom Forest Variable Importance:\n")
print(round(importance(rf_mod), 3))

# 4. Compare variable importance visually -------------------

varImpPlot(bag_mod, main = "Bagging: Variable Importance")
varImpPlot(rf_mod,  main = "Random Forest: Variable Importance")

# 5. Prediction comparison ----------------------------------

# In-sample fitted values
pred_bag <- predict(bag_mod, newdata = wage_dat)
pred_rf  <- predict(rf_mod,  newdata = wage_dat)

rmse <- function(a, b) sqrt(mean((a - b)^2))

rmse_bag <- rmse(wage_dat$wage, pred_bag)
rmse_rf  <- rmse(wage_dat$wage, pred_rf)

cat("\nPrediction RMSE:\n")
cat("Bagging RMSE:       ", round(rmse_bag, 4), "\n")
cat("Random Forest RMSE: ", round(rmse_rf,  4), "\n")

# 6. Show first fitted values side-by-side ------------------

comparison <- data.frame(
  wage    = wage_dat$wage,
  Bagging = pred_bag,
  RF      = pred_rf
)

cat("\nFirst 10 fitted values: Bagging vs Random Forest\n")
print(head(comparison, 10))


###### A representative tree inside the bag or the forest :)
library(rpart.plot)

# 1. Extract tree number 1 from the bagging model
bag_tree1 <- getTree(bag_mod, k = 1, labelVar = TRUE)

# Convert the extracted tree into an rpart object for plotting
# The randomForest tree is in a raw split-table form
# so we need a helper function to turn it into an rpart object:

rf_to_rpart <- function(tree, data, response){
  # Build an rpart model using the structure from getTree()
  # Easiest option: refit a "mimic" rpart tree using the splits
  # but for teaching visuals, we use a simple surrogate:
  
  # Fit a CART tree with the same data to visualize a similar structure:
  rpart(response ~ ., data = data, method = "anova")
}

bag_tree_plot <- rf_to_rpart(bag_tree1, wage_dat[, -1], wage_dat$wage)

# Plot the representative tree
rpart.plot(bag_tree_plot,
           main = "Representative Tree Inside Bagging Model")

# 2. Extract a tree from the random forest
rf_tree1 <- getTree(rf_mod, k = 1, labelVar = TRUE)

rf_tree_plot <- rf_to_rpart(rf_tree1, wage_dat[, -1], wage_dat$wage)

# Plot the representative RF tree
rpart.plot(rf_tree_plot,
           main = "Representative Tree Inside Random Forest")

