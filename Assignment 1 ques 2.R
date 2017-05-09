
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "Y")

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$Y = german_credit$Y - 1
credit.data = german_credit

set.seed(10676565)

subset <- sample(nrow(credit.data), nrow(credit.data) * 0.75)
credit.train = credit.data[subset, ]
credit.test = credit.data[-subset, ]

creditcost <- function(observed, predicted) {
  weight1 = 15
  weight0 = 1
  c1 = (observed == 1) & (predicted == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (observed == 0) & (predicted == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

# Generalized Linear Models (Logistic Regression)

credit.glm0 <- glm(Y ~ ., family = binomial, credit.train)
credit.glm.step <- step(credit.glm0, k = log(nrow(credit.train)), direction = c("both"))

summary(credit.glm.step)
AIC(credit.glm.step)
# AIC: 753.3981


library(glmnet)

prob.glm0.outsample <- predict(credit.glm.step, credit.train, type = "response")
predicted.glm0.outsample <- prob.glm0.outsample > 0.06
predicted.glm0.outsample <- as.numeric(predicted.glm0.outsample)
table(credit.train$Y, predicted.glm0.outsample, dnn = c("Truth", "Predicted"))

#Predicted
#Truth   0   1
#    0  98  416
#    1   4  232

mean(ifelse(credit.train$Y != predicted.glm0.outsample, 1, 0))
#0.56
creditcost(credit.train$Y, predicted.glm0.outsample)
# 0.63



prob.glm0.outsample <- predict(credit.glm.step, credit.test, type = "response")
predicted.glm0.outsample <- prob.glm0.outsample > 0.06
predicted.glm0.outsample <- as.numeric(predicted.glm0.outsample)
table(credit.test$Y, predicted.glm0.outsample, dnn = c("Truth", "Predicted"))

#       Predicted

# Truth   0    1
#    0   36  150
#    1    4   60

mean(ifelse(credit.test$Y != predicted.glm0.outsample, 1, 0))
#0.616
creditcost(credit.test$Y, predicted.glm0.outsample)
# 0.84
library("verification")
roc.plot(credit.test$Y == "1", prob.glm0.outsample)
roc.plot(credit.test$Y == "1", prob.glm0.outsample)$roc.vol

# Generalized Additive Models (GAM)

library(mgcv)

gam_formula <- as.formula(paste("Y~ s(age)+s(duration)+s(n_people)+ ",  paste(colnames(credit.train)[4:14], 
                                                                    collapse = "+")))

credit.gam <- gam(formula = gam_formula, family = binomial, data = credit.train)


summary(credit.gam)

plot(credit.gam, shade = TRUE, , seWithMean = TRUE, scale = 0)




AIC(credit.gam)
## [1] 1780
BIC(credit.gam)
## [1] 2190
credit.gam$deviance
## [1] 1659.40

pcut.gam <- 0.08
prob.gam.in <- predict(credit.gam, credit.train, type = "response")
pred.gam.in <- (prob.gam.in >= pcut.gam) * 1
table(credit.train$Y, pred.gam.in, dnn = c("Observation", "Prediction"))
#              Prediction
# Observation    0    1
#            0  3399  831
#            1   98  172
mean(ifelse(credit.train$Y != pred.gam.in, 1, 0))

# 0.2064

AIC(credit.gam)
## [1] 1780
BIC(credit.gam)
## [1] 2190

# define the searc grid from 0.01 to 0.20
searchgrid = seq(0.01, 0.2, 0.01)
# result.gam is a 99x2 matrix, the 1st col stores the cut-off p, the 2nd
# column stores the cost
result.gam = cbind(searchgrid, NA)
# in the cost function, both r and pi are vectors, r=truth, pi=predicted
# probability
cost1 <- function(r, pi) {
  weight1 = 15
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

for (i in 1:length(searchgrid)) {
  pcut <- result.gam[i, 1]
  # assign the cost to the 2nd col
  result.gam[i, 2] <- cost1(credit.train$Y, predict(credit.gam, type = "response"))
}
plot(result.gam, ylab = "Cost in Training Set")

index.min <- which.min(result.gam[, 2])  #find the index of minimum value
result.gam[index.min, 2]  #min cost
##        
## 0.5
result.gam[index.min, 1]  #optimal cutoff probability
## searchgrid 
##       0.06
#Insample
pcut <- result.gam[index.min, 1]

prob.gam.out <- predict(credit.gam, credit.train, type = "response")

pred.gam.out <- (prob.gam.out >= pcut) * 1

table(credit.train$Y, pred.gam.out, dnn = c("Observation", "Prediction"))

##            Prediction
## Observation   0   1
##           0 3077  1153
##           1   73   197

# mis-classifciation rate is

mean(ifelse(credit.train$Y != pred.gam.out, 1, 0))
## [1] 0.272

#Cost associated with misclassification is

creditcost(credit.train$Y, pred.gam.out)
## [1] 0.5

# Out-of-sample fit performance

pcut <- result.gam[index.min, 1]

prob.gam.out <- predict(credit.gam, credit.test, type = "response")

pred.gam.out <- (prob.gam.out >= pcut) * 1

table(credit.test$Y, pred.gam.out, dnn = c("Observation", "Prediction"))

##            Prediction
## Observation   0   1
##           0 341  129
##           1  9   21

# mis-classifciation rate is

mean(ifelse(credit.test$Y != pred.gam.out, 1, 0))
## [1] 0.276

#Cost associated with misclassification is

creditcost(credit.test$Y, pred.gam.out)
## [1] 0.528


# Discriminant Analysis


# Insample

credit.train$Y = as.factor(credit.train$Y)
credit.lda <- lda(Y ~ ., data = credit.train)
prob.lda.in <- predict(credit.lda, data = credit.train)
pcut.lda <- 0.06
pred.lda.in <- (prob.lda.in$posterior[, 2] >= pcut.lda) * 1
table(credit.train$Y, pred.lda.in, dnn = c("Obs", "Pred"))

#         Pred
# Obs    0    1
# 0   3186  1044
# 1     87   183

# Outsample


mean(ifelse(credit.train$Y != pred.lda.in, 1, 0))

# 0.25133

creditcost(credit.train$Y, pred.lda.in)

# 0.522

lda.out <- predict(credit.lda, newdata = credit.test)
cut.lda <- 0.06
pred.lda.out <- as.numeric((lda.out$posterior[, 2] >= cut.lda))
table(credit.test$Y, pred.lda.out, dnn = c("Obs", "Pred"))
#  Pred
#  Obs   0   1
#  0   356  114
#  1     8   22
 
mean(ifelse(credit.test$Y != pred.lda.out, 1, 0))

# 0.244

creditcost(credit.test$Y, pred.lda.out)
 
# 0.468

# Tree
library(rpart)
credit.rpart <- rpart(formula = Y ~ . , data = credit.train, method = "class", 
                      parms = list(loss = matrix(c(0, 15, 1, 0), nrow = 2)))
plot(credit.rpart)
text(credit.rpart)

boston.largetree <-  rpart(formula = Y ~ . , data = credit.train, method = "class", 
                           parms = list(loss = matrix(c(0, 15, 1, 0), nrow = 2)),cp=0.001)



plotcp(boston.largetree)


credit.test.pred.tree1 = predict(credit.rpart, credit.train, type = "class")
table(credit.train$Y, credit.test.pred.tree1, dnn = c("Truth", "Predicted"))


credit.test.pred.tree1 = predict(credit.rpart, credit.test, type = "class")
table(credit.test$Y, credit.test.pred.tree1, dnn = c("Truth", "Predicted"))






