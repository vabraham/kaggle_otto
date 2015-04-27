# Clear environment workspace
rm(list=ls())
# Load data
train <- read.csv("/Users/vabraham24/Documents/RStudio/kaggle_otto/data/train.csv")
test <- read.csv("/Users/vabraham24/Documents/RStudio/kaggle_otto/data/test.csv")
samplesub <- read.csv("/Users/vabraham24/Documents/RStudio/kaggle_otto/data/sampleSubmission.csv")
# Remove id column so it doesn't get picked up by the current classifier
train <- train[,-1]
test <- test[,-1]
summary(train)
summary(test)
# Create sample train and test datasets for prototyping new models.
strain <- train[sample(nrow(train), 6000, replace=FALSE),]
stest <- train[sample(nrow(train), 2000, replace=FALSE),]

# Install tree package
install.packages('tree')
library(tree)
# Set a unique seed number so you get the same results everytime you run the below model,
# The number does not matter
set.seed(12)
# Create a decision tree model using the target field as the response and all 93 features as inputs
fit1 <- tree(as.factor(target) ~ ., data=strain)
plot(fit1)
title(main="tree")
text(fit1)
# Test the tree model on the holdout test dataset
fit1.pred <- predict(fit1, stest, type="class")
table(fit1.pred,stest$target)
fit1$error <- 1-(sum(fit1.pred==stest$target)/length(stest$target))
fit1$error

# The dec. tree model from the tree package didn't perform very well, let's try the dec. tree model from the rpart package
# Install rpart package
install.packages('rpart')
library(rpart)
# set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(13)
fit2 <- rpart(as.factor(target) ~ ., data=train, method="class")
par(xpd=TRUE)
plot(fit2, compress=TRUE)
title(main="rpart")
text(fit2)
# Test the rpart (tree) model on the holdout test dataset
fit2.pred <- predict(fit2, stest, type="class")
table(fit2.pred,stest$target)
fit2$error <- 1-(sum(fit2.pred==stest$target)/length(stest$target))
fit2$error

# Neither of the decision tree models are working very well, not all of the classes are being predicted. This is probably
# because there are too many features that recursive binary partitioning is over-simplifying the model and missing some of 
# the classes completely.

# Install adabag package
install.packages('adabag')
library(adabag)
# set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(14)
# Run the standard version of the bagging model.
ptm3 <- proc.time()
fit3 <- bagging(target ~ ., data=strain, mfinal=50)
fit3$time <- proc.time() - ptm3
# Test the baggind model on the holdout test dataset
fit3.pred <- predict(fit3, stest, newmfinal=50)
table(as.factor(fit3.pred$class),stest$target)
fit3$error

# Install randomForest package
install.packages('randomForest')
library(randomForest)
# set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(16)
# Use the tuneRF function to determine an ideal value for the mtry parameter
mtry <- tuneRF(strain[,1:93], strain[,94], mtryStart=1, ntreeTry=50, stepFactor=2, improve=0.05,
               trace=TRUE, plot=TRUE, doBest=FALSE)
# The ideal mtry value was found to be 8
# Create a random forest model using the target field as the response and all 93 features as inputs
ptm4 <- proc.time()
fit4 <- randomForest(as.factor(target) ~ ., data=strain, importance=TRUE, ntree=100, mtry=8)
fit4.time <- proc.time() - ptm4
# Create a dotchart of variable/feature importance as measured by a Random Forest
varImpPlot(fit4)
# Test the randomForest model on the holdout test dataset
fit4.pred <- predict(fit4, stest, type="response")
table(fit4.pred,stest$target)
fit4$error <- 1-(sum(fit4.pred==stest$target)/length(stest$target))
fit4$error

# Install gbm package
install.packages('gbm')
library(gbm)
# set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(17)
ptm5 <- proc.time()
fit5 <- gbm(target ~ ., data=strain, distribution="multinomial", n.trees=2000, cv.folds=2)
fit5.time <- proc.time() - ptm5
trees <- gbm.perf(fit5)
# Test the gbm model on the holdout test dataset
fit5.stest <- predict(fit5, stest, n.trees=trees, type="response")
fit5.stest <- as.data.frame(fit5.stest)
names(fit5.stest) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
fit5.stest.pred <- rep(NA,2000)
for (i in 1:nrow(stest)) {
  fit5.stest.pred[i] <- colnames(fit5.stest)[(which.max(fit5.stest[i,]))]}
fit5.pred <- as.factor(fit5.stest.pred)
table(fit5.pred,stest$target)
fit5.pred <- as.character(fit5.pred)
fit5$error <- 1-(sum(fit5.pred==stest$target)/length(stest$target))
fit5$error