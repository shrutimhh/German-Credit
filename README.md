# German-Credit

#############Start of the Code ###############

german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", "present_resid", "property", "age", "other_install", "housing", "n_credits", "job", "n_people", "telephone", "foreign", "response")

#original response coding 1= good, 2 = bad we need 0 = good, 1 = bad

german_credit$response = german_credit$response - 1

#############End of the Code ###############


# for 80% training set
index <- sample(nrow(german_credit),nrow(german_credit)*0.80)
credit_train = german_credit[index,]
credit_test = german_credit[-index,]

credit_train
summary(credit_train) #for EDA

library(ggplot2) #for box plots
boxplot(credit_train[,2])
boxplot(credit_train[,5])
boxplot(credit_train[,8])
boxplot(credit_train[,11])
boxplot(credit_train[,13])
boxplot(credit_train[,16])
boxplot(credit_train[,18])

credit_glm0 <- glm(response~., family=binomial, data=credit_train)
credit_glm0 # for the logistic regression model
summary(credit_glm0) 

AIC(credit_glm0) # to evaluate model fitting

hist(predict(credit_glm0)) # histogram
pred_resp <- predict(credit_glm0,type="response")
hist(pred_resp)

# to find best model
nullmodel = lm(response ~ 1, data = credit_train)
fullmodel = lm(response ~ ., data = credit_train)
model.step <- step(nullmodel, scope = list(lower = nullmodel,
                                           upper = fullmodel),
                   direction = "forward")

# best model
model_opt <- glm(formula = response ~ chk_acct + duration + credit_his + purpose + other_debtor + 
                   saving_acct + installment_rate + housing + amount + foreign + 
                   telephone + age, family = binomial, data = credit_train)
summary(model_opt) 
AIC(model_opt) # aic for optimal model

install.packages('ROCR')
library(ROCR) # In sample prediction
pred_glm0_train <- predict(model_opt, type = "response")
pred <- prediction(pred_glm0_train, credit_train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE) # ROC curve for training set

# Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

# for the misclassification table
table(credit_train$response, (pred_glm0_train > 0.5)*1, dnn=c("Truth","Predicted"))

costfunc <- function(obs, pred.p){
  weight1 <- 5 # define the weight for "true=1 but pred=0" (FN)
  weight0 <- 1 # define the weight for "true=0 but pred=1" (FP)
  pcut <- 0.5
  c1 <- (obs==1)&(pred.p < pcut) # count for "true=1 but pred=0" (FN)
  c0 <- (obs==0)&(pred.p >= pcut) # count for "true=0 but pred=1" (FP)
  cost <- mean(weight1*c1 + weight0*c0) # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
}

# out of sample prediction
pred_glm0_test<- predict(credit_glm0, newdata = credit_test, type="response")
pred <- prediction(pred_glm0_test, credit_test$response) 
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)  

# Get the AUC for out of sample
unlist(slot(performance(pred, "auc"), "y.values"))

# for 5 fold cross validation
library(boot)
credit_glm1<- glm(response~. , family=binomial, data=german_credit);
cv_result <- cv.glm(data=german_credit, glmfit=credit_glm1, cost=costfunc, K=
                      5)
cv_result$delta[2]



### for repeating above with new 80% training data
# for 80% training set
index2 <- sample(nrow(german_credit),nrow(german_credit)*0.80)
credit_train2 = german_credit[index2,]
credit_test2 = german_credit[-index2,]

credit_train2
summary(credit_train2) #for EDA

library(ggplot2) #for boxplots
boxplot(credit_train2[,2])
boxplot(credit_train2[,5])
boxplot(credit_train2[,8])
boxplot(credit_train2[,11])
boxplot(credit_train2[,13])
boxplot(credit_train2[,16])
boxplot(credit_train2[,18])

credit_glm02 <- glm(response~., family=binomial, data=credit_train2)
credit_glm02 # for the logistic regression model
summary(credit_glm02) 

AIC(credit_glm02) # to evaluate model fitting

hist(predict(credit_glm02)) #histogram
pred_resp2 <- predict(credit_glm02,type="response")
hist(pred_resp2)

# to find best model
nullmodel2 = lm(response ~ 1, data = credit_train2)
fullmodel2 = lm(response ~ ., data = credit_train2)
model.step <- step(nullmodel2, scope = list(lower = nullmodel2,
                                            upper = fullmodel2),
                   direction = "forward")

# best model
model_opt2 <- glm(formula = response ~ chk_acct + duration + credit_his + purpose + present_emp + 
                    foreign + saving_acct + other_debtor + age + housing + other_install + 
                    amount + installment_rate, family = binomial, data = credit_train2)
summary(model_opt2) 
AIC(model_opt2) #AIC for optimal model

install.packages('ROCR')
library(ROCR) #In sample prediction
pred_glm0_train2 <- predict(model_opt2, type = "response")
pred2 <- prediction(pred_glm0_train2, credit_train2$response)
perf2 <- performance(pred2, "tpr", "fpr")
plot(perf2, colorize=TRUE) #ROC curve for training set

# Get the AUC
unlist(slot(performance(pred2, "auc"), "y.values"))

# for the misclassification table
table(credit_train2$response, (pred_glm0_train2 > 0.5)*1, dnn=c("Truth","Predicted"))

costfunc2 <- function(obs, pred.p){
  weight1 <- 5 # define the weight for "true=1 but pred=0" (FN)
  weight0 <- 1 # define the weight for "true=0 but pred=1" (FP)
  pcut <- 0.5
  c12 <- (obs==1)&(pred.p < pcut) # count for "true=1 but pred=0" (FN)
  c02 <- (obs==0)&(pred.p >= pcut) # count for "true=0 but pred=1" (FP)
  cost2 <- mean(weight1*c12 + weight0*c02) # misclassification with weight
  return(cost2) # you have to return to a value when you write R functions
}

# out of sample prediction
pred_glm0_test2<- predict(credit_glm02, newdata = credit_test2, type="response")
pred2 <- prediction(pred_glm0_test2, credit_test2$response) 
perf2 <- performance(pred2, "tpr", "fpr")
plot(perf2, colorize=TRUE)  

# Get the AUC for out of sample
unlist(slot(performance(pred2, "auc"), "y.values"))

# for 5 fold cross validation
library(boot)
credit_glm12<- glm(response~. , family=binomial, data=german_credit);
cv_result2 <- cv.glm(data=german_credit, glmfit=credit_glm12, cost=costfunc2, K=
                       5)
cv_result2$delta[2]
